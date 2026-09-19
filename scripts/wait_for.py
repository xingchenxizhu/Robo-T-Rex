"""等待某个产物出现（用于长时间阻塞等待，避免反复轮询）。

    .venv\\Scripts\\python.exe scripts\\wait_for.py runs/s3_loco/final_model.zip [more...]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

targets = []
for a in sys.argv[1:]:
    p = Path(a)
    targets.append(p if p.is_absolute() else ROOT / p)
if not targets:
    print("用法: wait_for.py <path> [path...]")
    raise SystemExit(2)

start = time.time()
missing = [str(t.relative_to(ROOT)) for t in targets]
print("等待:", ", ".join(missing), flush=True)
while True:
    if all(t.exists() for t in targets):
        print(f"[{time.time()-start:6.1f}s] 全部出现: " +
              ", ".join(str(t.relative_to(ROOT)) for t in targets), flush=True)
        break
    time.sleep(15)
