"""阶段完成通知器：只在阶段切换/完成时输出一行，便于长时间阻塞等待。

    .venv\\Scripts\\python.exe scripts\\notify_stages.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

ORDER = ["s1_stand", "s2_reach", "s3_loco", "s4_chase"]


def snapshot():
    out = {}
    for name in ORDER:
        d = ROOT / "runs" / name
        if not d.exists():
            continue
        meta = {}
        if (d / "train_meta.json").exists():
            try:
                meta = json.loads((d / "train_meta.json").read_text(encoding="utf-8"))
            except Exception:  # noqa: BLE001
                meta = {}
        done = 0
        ckdir = d / "checkpoints"
        if ckdir.exists():
            for c in ckdir.glob("*.zip"):
                try:
                    done = max(done, int(c.stem.split("_")[1]))
                except Exception:  # noqa: BLE001
                    pass
        out[name] = {
            "task": meta.get("task"),
            "plan": meta.get("steps"),
            "done": done,
            "finished": bool(meta.get("wall_seconds")),
            "has_final": (d / "final_model.zip").exists(),
        }
    return out


prev = {}
start = time.time()
print(f"[{0:6.1f}s] notifier 启动", flush=True)
while True:
    snap = snapshot()
    for name in ORDER:
        cur = snap.get(name)
        old = prev.get(name)
        if cur and cur != old:
            tag = "完成" if cur["finished"] else "训练中"
            print(f"[{time.time()-start:6.1f}s] {name} ({cur['task']}) "
                  f"plan={cur['plan']} done>={cur['done']} {tag}", flush=True)
    prev = snap
    if snap and all(s["finished"] and s["has_final"] for s in snap.values()) \
            and len(snap) == len(ORDER):
        print(f"[{time.time()-start:6.1f}s] 四个阶段全部完成", flush=True)
        break
    time.sleep(30)
