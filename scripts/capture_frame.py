"""从正在运行的查看器抓一帧，报告亮度并保存成图片（用于确认视口不是全黑）。

    .venv\\Scripts\\python.exe scripts\\capture_frame.py --port 8770 `
        --task loco --mode active --checkpoint runs/s3_loco/selected_model.zip `
        --out preview/ui_capture.png --steps 240
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request as urlreq
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--task", default="loco")
    ap.add_argument("--mode", default="active")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--out", default="preview/ui_capture.jpg",
                    help="服务端返回的是 JPEG，扩展名要与之一致")
    ap.add_argument("--sim-seconds", type=float, default=0.0,
                    help="先把仿真推进到该时刻（>0 时生效；服务端按真实耗时推进，需小睡）")
    ap.add_argument("--steps", type=int, default=240, help="先推进多少控制步再抓图")
    ap.add_argument("--speed", type=float, default=1.0)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    def get(p):
        with urlreq.urlopen(base + p, timeout=120) as r:
            return json.loads(r.read())

    def post(p, payload):
        req = urlreq.Request(base + p, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
        with urlreq.urlopen(req, timeout=180) as r:
            return json.loads(r.read())

    d = post("/api/session", {"task": args.task, "mode": args.mode,
                              "checkpoint": args.checkpoint})
    if "error" in d:
        print("会话创建失败:", d["error"])
        return 1
    sid = d["sid"]
    print(f"会话 {sid}  obs={d['state']['obs_dim']}  策略={d['state']['has_policy']}")

    if args.sim_seconds > 0:
        import time as _t

        deadline = _t.time() + 60
        while _t.time() < deadline:
            f = get(f"/api/frame?sid={sid}&play=1&speed=1.0")
            if "error" in f:
                print("推进失败:", f["error"])
                return 1
            if f["state"]["time"] >= args.sim_seconds or f["state"]["fallen"]:
                break
            _t.sleep(0.04)
    f = get(f"/api/frame?sid={sid}&play=0&speed=0")
    if "error" in f:
        print("取帧失败:", f["error"])
        return 1
    img = base64.b64decode(f["frame"])
    out = ROOT / args.out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_bytes(img)
    from PIL import Image

    im = Image.open(io.BytesIO(img)).convert("L")
    px = list(im.getdata())
    mean = sum(px) / len(px)
    st = f["state"]
    print(f"画面: {im.size} 平均亮度={mean:.2f} 峰值={max(px)}")
    print(f"状态: 任务={st['task']} 模式={st['mode']} 阶段={st['phase']} "
          f"t={st['time']:.2f}s 速度={st['speed']:.3f} 步态={st['gait']}")
    print("[written]", out)
    return 0 if mean > 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
