"""验证交互式查看器是否可用（需要先启动 app.py）。

    .venv\\Scripts\\python.exe scripts\\verify_ui.py [--port 8770]
"""
from __future__ import annotations

import argparse
import json
import urllib.request as urlreq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8770)
    args = ap.parse_args()
    base = f"http://127.0.0.1:{args.port}"

    def get(path):
        with urlreq.urlopen(base + path, timeout=90) as r:
            return json.loads(r.read())

    def post(path, payload):
        req = urlreq.Request(base + path, data=json.dumps(payload).encode(),
                             headers={"Content-Type": "application/json"})
        with urlreq.urlopen(req, timeout=120) as r:
            return json.loads(r.read())

    print("GET /            ->", urlreq.urlopen(base + "/", timeout=30).status)
    runs = get("/api/runs")["runs"]
    print("runs             ->", [r["name"] for r in runs])
    print("comparisons      ->", [c["task"] for c in get("/api/compare")["comparisons"]])

    cases = [("stand", "active", "runs/s1_stand/final_model.zip"),
             ("reach", "active", "runs/s2_reach/final_model.zip"),
             ("loco", "active", None),
             ("chase", "active", "runs/s4_chase/final_model.zip"),
             ("stand", "passive", "runs/s1_stand/final_model.zip"),
             ("stand", "fixed", "runs/s1_stand/final_model.zip")]
    for task, mode, ckpt in cases:
        d = post("/api/session", {"task": task, "mode": mode, "checkpoint": ckpt})
        if "error" in d:
            print(f"  {task:6s}/{mode:7s} FAIL {d['error']}")
            continue
        sid = d["sid"]
        st = d["state"]
        f = get(f"/api/frame?sid={sid}&play=1&speed=1.0")
        st2 = f["state"]
        print(f"  {task:6s}/{mode:7s} obs={st['obs_dim']:3d} policy={str(st['has_policy']):5s} "
              f"jpeg={len(f['frame']):6d}B  phase={st2['phase']:8s} t={st2['time']:.2f}s "
              f"gait={st2['gait']}")
    tb = get("/api/tb?run=s3_loco")["series"]
    print("tb 标签数         ->", len(tb), "例:", sorted(tb)[:3])
    print("OK" if tb else "TB 无数据")


if __name__ == "__main__":
    main()
