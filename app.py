"""机器霸王龙交互式查看器（Flask + MuJoCo 离屏渲染）。

启动：
    .venv\\Scripts\\python.exe app.py --port 8770
然后打开 http://127.0.0.1:8770

功能：
  * 选择实验（runs/*）、检查点（selected_model 优先推荐，另有 best / final / 各 ckpt）
  * 播放 / 暂停 / 单步 / 重置，0.05x ~ 2x 慢放
  * 切换任务（stand / reach / loco / chase）与尾巴对照（active / passive / fixed）
  * 叠加扰动开关
  * 实时显示姿态、足部接触、接触点、目标位置、尾关节角度条
  * 读取 TensorBoard 事件，画训练进度 / 奖励 / 成功率 / 晃动 / 尾巴行程曲线

两个**必须保留**的实现约束（改动前请先读这段）：

1. `app.run(..., threaded=False)`。MuJoCo 的 OpenGL 上下文是**线程绑定**的：
   多线程下渲染器会在不同线程间失效，接口返回全黑帧
   （日志表现为 `GLFWError: WGL: Failed to make context current`）。
   同理，渲染器要在 `Session.__init__` 里**先渲一帧**，把上下文建在处理请求的线程上。
2. 允许多个会话并存（`MAX_SESSIONS`，超出淘汰最旧的）。
   早期版本只保留一个会话，导致"另一个页面或脚本一建会话就把当前页面的会话顶掉"，
   页面只能拿到 404、视口停在空 `<img>` 上显示全黑。
   前端在遇到会话失效时会自动重建（见 `web/index.html` 的 `tick()`）。

自检：`scripts/verify_ui.py`（各任务/模式全链路）、
`scripts/stress_sessions.py`（多会话渲染不出黑帧）、
`scripts/capture_frame.py`（抓一帧存图并报告亮度）。
"""

from __future__ import annotations

import argparse
import base64
import io
import json
import threading
import time
from pathlib import Path

from flask import Flask, jsonify, request, send_from_directory
from PIL import Image

ROOT = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(ROOT))

from trex import TrexEnv  # noqa: E402

app = Flask(__name__, static_folder=None)
LOCK = threading.Lock()
SESSIONS: dict[str, "Session"] = {}
MAX_SESSIONS = 8
RUNS_DIR = ROOT / "runs"

TASKS = ("stand", "reach", "loco", "chase")
MODES = ("active", "passive", "fixed")


# --------------------------------------------------------------------------- #
class Session:
    def __init__(self, checkpoint, task, mode, perturb=False, curriculum=0, seed=0,
                 speed_scale=None, target_randomize=None):
        self.task, self.mode = task, mode
        self.env = TrexEnv(task=task, mode=mode, seed=seed, perturb=perturb,
                           curriculum=curriculum, record_trace=True)
        if speed_scale is not None:
            self.env.speed_scale = float(speed_scale)
        if target_randomize is not None:
            self.env.target_randomize = float(target_randomize)
        self.model = None
        self.checkpoint = checkpoint
        if checkpoint:
            from stable_baselines3 import PPO

            path = Path(checkpoint)
            if not path.is_absolute():
                path = ROOT / path
            if path.suffix != ".zip" and path.with_suffix(".zip").exists():
                path = path.with_suffix(".zip")
            self.model = PPO.load(str(path), env=self.env, device="cpu")
            want = int(self.env.observation_space.shape[0])
            got = int(self.model.observation_space.shape[0])
            if want != got:
                raise ValueError(
                    f"检查点观测维度 {got} 与当前任务 {task} 的 {want} 不匹配，"
                    f"请选择与该任务同阶段的实验")
        self.obs, _ = self.env.reset(seed=seed)
        # 立刻渲染一次：MuJoCo 的 OpenGL 上下文是**线程绑定**的，
        # 必须在处理请求的同一个线程里创建；否则后续请求会拿到全黑帧
        # （日志表现为 GLFWError: WGL: Failed to make context current）。
        self.env.render()
        self.last_t = time.perf_counter()
        self.total_sim_time = 0.0
        self.episodes = 0
        self.episode_returns = []
        self._ret = 0.0

    # ---------------------------------------------------------------- #
    def reset(self, seed=None):
        self.obs, _ = self.env.reset(seed=seed)
        self.episodes += 1
        if self._ret:
            self.episode_returns.append(self._ret)
        self._ret = 0.0

    def advance(self, sim_seconds: float):
        n = int(round(sim_seconds / self.env.dt))
        n = max(1, min(n, 200))
        last_info = None
        for _ in range(n):
            if self.model is None:
                action = self._stand_action()
            else:
                action, _ = self.model.predict(self.obs, deterministic=True)
            self.obs, r, term, trunc, info = self.env.step(action)
            self._ret += float(r)
            self.total_sim_time += self.env.dt
            last_info = info
            if term or trunc:
                self.episodes += 1
                self.episode_returns.append(self._ret)
                self._ret = 0.0
                self.obs, _ = self.env.reset()
        return last_info

    def _stand_action(self):
        """无检查点时给一个保姿态的零动作，便于查看纯物理形态差异。"""
        import numpy as np

        return np.zeros(self.env.action_space.shape, dtype=np.float32)

    def frame(self):
        img = self.env.render()
        buf = io.BytesIO()
        Image.fromarray(img).save(buf, format="JPEG", quality=78)
        return base64.b64encode(buf.getvalue()).decode("ascii")

    def state(self):
        st = self.env.state()
        st["episodes"] = self.episodes
        st["ep_return"] = self._ret
        st["recent_returns"] = self.episode_returns[-20:]
        st["obs_dim"] = int(self.env.observation_space.shape[0])
        st["has_policy"] = self.model is not None
        return st


# --------------------------------------------------------------------------- #
def scan_runs():
    out = []
    if not RUNS_DIR.exists():
        return out
    for d in sorted(RUNS_DIR.iterdir()):
        if not d.is_dir():
            continue
        meta_path = d / "train_meta.json"
        meta = {}
        if meta_path.exists():
            try:
                meta = json.loads(meta_path.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = {}
        ckpts = []
        # 优先推荐按确定性评估选出的检查点（select_ckpt.py 的产物）：
        # 训练期的 best_model 可能停在难度较低、或指标被探索噪声污染的阶段。
        for name in ("selected_model.zip", "best_model.zip", "final_model.zip"):
            if (d / name).exists():
                ckpts.append(name)
        ckdir = d / "checkpoints"
        if ckdir.exists():
            ckpts += [f"checkpoints/{p.name}" for p in sorted(ckdir.glob("*.zip"))]
        metrics = {}
        if (d / "metrics.json").exists():
            try:
                metrics = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                metrics = {}
        out.append({
            "name": d.name,
            "task": meta.get("task"),
            "mode": meta.get("mode"),
            "steps": meta.get("steps"),
            "wall_seconds": meta.get("wall_seconds"),
            "checkpoints": ckpts,
            "best_score": metrics.get("best_score"),
            "episodes": metrics.get("episodes"),
            "has_tb": (d / "tb").exists(),
        })
    return out


def read_tb(run_name: str, max_points: int = 240):
    """读取 TensorBoard 事件里的标量，降采样后返回。"""
    run_dir = RUNS_DIR / run_name / "tb"
    if not run_dir.exists():
        return {}
    from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

    series: dict[str, list] = {}
    for sub in sorted(run_dir.iterdir()):
        if not sub.is_dir():
            continue
        try:
            acc = EventAccumulator(str(sub), size_guidance={"scalars": 0})
            acc.Reload()
        except Exception:  # noqa: BLE001
            continue
        for tag in acc.Tags().get("scalars", []):
            events = acc.Scalars(tag)
            series.setdefault(tag, [])
            series[tag] += [[e.step, float(e.value)] for e in events]
    out = {}
    for tag, pts in series.items():
        pts.sort(key=lambda x: x[0])
        if len(pts) > max_points:
            step = len(pts) / max_points
            pts = [pts[min(len(pts) - 1, int(i * step))] for i in range(max_points)]
        out[tag] = pts
    return out


# --------------------------------------------------------------------------- #
@app.route("/")
def index():
    return send_from_directory(str(ROOT / "web"), "index.html")


@app.route("/api/runs")
def api_runs():
    return jsonify({"runs": scan_runs()})


@app.route("/api/tb")
def api_tb():
    name = request.args.get("run", "")
    return jsonify({"run": name, "series": read_tb(name)})


@app.after_request
def _no_cache(resp):
    """前端与接口都不缓存：改了 index.html / app.py 后刷新即可生效，
    否则浏览器会拿着旧的 index.html 继续用已经失效的会话 id。"""
    resp.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    resp.headers["Pragma"] = "no-cache"
    return resp


@app.route("/api/session", methods=["POST"])
def api_session():
    """创建会话。支持多个会话并存（最多 MAX_SESSIONS 个，超出淘汰最旧的）。

    早期版本只保留一个会话，结果"另一个页面/脚本一建会话"就把当前页面的
    会话顶掉，页面只能拿到 404、视口停在空 <img> 上显示全黑。
    """
    body = request.get_json(force=True) or {}
    sid = f"s{int(time.time()*1000)}"
    with LOCK:
        while len(SESSIONS) >= MAX_SESSIONS:
            oldest = next(iter(SESSIONS))
            try:
                SESSIONS[oldest].env.close()
            except Exception:  # noqa: BLE001
                pass
            SESSIONS.pop(oldest, None)
        try:
            sess = Session(
                checkpoint=body.get("checkpoint"),
                task=body.get("task", "stand"),
                mode=body.get("mode", "active"),
                perturb=bool(body.get("perturb", False)),
                curriculum=int(body.get("curriculum", 0)),
                seed=int(body.get("seed", 0)),
                speed_scale=body.get("speed_scale"),
                target_randomize=body.get("target_randomize"),
            )
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"{type(exc).__name__}: {exc}"}), 400
        SESSIONS[sid] = sess
        state = sess.state()
    return jsonify({"sid": sid, "state": state, "sessions": len(SESSIONS)})


@app.route("/api/reset", methods=["POST"])
def api_reset():
    sid = request.args.get("sid", "")
    with LOCK:
        sess = SESSIONS.get(sid)
        if sess is None:
            return jsonify({"error": "no session"}), 404
        sess.reset()
        return jsonify({"state": sess.state()})


@app.route("/api/frame")
def api_frame():
    sid = request.args.get("sid", "")
    speed = float(request.args.get("speed", 1.0))
    play = request.args.get("play", "0") == "1"
    with LOCK:
        sess = SESSIONS.get(sid)
        if sess is None:
            return jsonify({"error": "会话已失效，请重新加载"}), 404
        now = time.perf_counter()
        elapsed = min(0.3, now - sess.last_t)
        sess.last_t = now
        if play:
            sess.advance(max(0.0, elapsed) * speed)
        try:
            frame = sess.frame()
        except Exception as exc:  # noqa: BLE001
            return jsonify({"error": f"渲染失败: {type(exc).__name__}: {exc}"}), 500
        return jsonify({"frame": frame, "state": sess.state()})


@app.route("/api/compare")
def api_compare():
    """列出已生成的对照实验文件。"""
    out = []
    base = ROOT / "evaluations"
    if base.exists():
        for p in sorted(base.rglob("compare.json")):
            try:
                data = json.loads(p.read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                continue
            out.append({"path": str(p.relative_to(ROOT)), "task": data.get("task"),
                        "model": data.get("model"),
                        "summary": data.get("summary", {})})
    return jsonify({"comparisons": out})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    print(f"机器霸王龙查看器: http://{args.host}:{args.port}")
    # threaded=False 是刻意的：MuJoCo 的 OpenGL 上下文线程绑定，
    # 多线程会让渲染器在不同线程里失效并返回全黑帧。
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=False,
            use_reloader=False)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
