"""多会话压力检查：同时开多个会话并交替取帧，确认每一帧都不是全黑。

验证的假设：MuJoCo 的 OpenGL 上下文在**单线程**下可以多实例共存
（Flask 用 threaded=False），因此不需要"只保留一个会话"这种会顶掉
其它页面的设计。

    .venv\\Scripts\\python.exe scripts\\stress_sessions.py --port 8770
"""
from __future__ import annotations

import argparse
import base64
import io
import json
import urllib.request as urlreq


def brightness(b64: str) -> float:
    from PIL import Image

    img = Image.open(io.BytesIO(base64.b64decode(b64))).convert("L")
    px = list(img.getdata())
    return sum(px) / len(px)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    ap.add_argument("--rounds", type=int, default=3)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    def get(p):
        with urlreq.urlopen(base + p, timeout=180) as r:
            return json.loads(r.read())

    def post(p, payload):
        req = urlreq.Request(base + p, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
        with urlreq.urlopen(req, timeout=180) as r:
            return json.loads(r.read())

    cases = [("stand", "active", "runs/s1_stand/selected_model.zip"),
             ("reach", "active", "runs/s2_reach/selected_model.zip"),
             ("loco", "active", "runs/s3_loco/selected_model.zip"),
             ("chase", "active", "runs/s4_chase/selected_model.zip")]

    def make(task, mode, ck):
        d = post("/api/session", {"task": task, "mode": mode, "checkpoint": ck})
        if "error" in d:
            print(f"  建会话失败 {task}: {d['error']}")
            return None
        print(f"  已建会话 {task}: {d['sid']}  并存会话数={d.get('sessions')}")
        return d["sid"]

    sids = [(t, make(t, m, c)) for t, m, c in cases]
    if any(s is None for _, s in sids):
        return 1

    worst = 999.0
    evicted_total = 0
    for rnd in range(args.rounds):
        for i, (task, sid) in enumerate(list(sids)):
            f = get(f"/api/frame?sid={sid}&play=1&speed=1.0")
            if "error" in f:
                # 被淘汰（其它客户端建了更多会话）时重建，不算失败
                evicted_total += 1
                sids[i] = (task, make(task, cases[i][1], cases[i][2]))
                f = get(f"/api/frame?sid={sids[i][1]}&play=1&speed=1.0")
                if "error" in f:
                    print(f"  第{rnd+1}轮 {task}: 重建后仍取帧失败 {f['error']}")
                    return 1
            b = brightness(f["frame"])
            worst = min(worst, b)
            print(f"  第{rnd+1}轮 {task:6s} 亮度={b:6.2f}  t={f['state']['time']:.2f}s "
                  f"步态={f['state']['gait']}")
    print(f"\n最小亮度 = {worst:.2f}   会话被淘汰次数 = {evicted_total}")
    print("多会话渲染正常，没有全黑帧" if worst > 3 else "存在全黑帧，多上下文仍有问题")
    return 0 if worst > 3 else 2


if __name__ == "__main__":
    raise SystemExit(main())
