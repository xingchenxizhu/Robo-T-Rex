"""训练进度速览（一行一个阶段）。

    .venv\\Scripts\\python.exe scripts\\watch.py           # 紧凑
    .venv\\Scripts\\python.exe scripts\\watch.py -v        # 详细
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from app import read_tb  # noqa: E402

ORDER = ["s1_stand", "s2_reach", "s3_loco", "s4_chase"]
DETAIL = ["rollout/ep_rew_mean", "rollout/ep_len_mean", "time/fps",
          "trex/success_rate", "trex/fall_rate", "trex/com_sway",
          "trex/tail_rom_mean", "trex/tail_momentum_share", "train/std",
          "trex/success_locomotion", "trex/success_run", "trex/reached",
          "trex/bitten", "trex/extra_steps", "trex/recovery_s"]
verbose = "-v" in sys.argv

for name in ORDER:
    d = ROOT / "runs" / name
    if not d.exists():
        continue
    meta = {}
    if (d / "train_meta.json").exists():
        meta = json.loads((d / "train_meta.json").read_text(encoding="utf-8"))
    done = 0
    ckdir = d / "checkpoints"
    if ckdir.exists():
        for c in ckdir.glob("*.zip"):
            try:
                done = max(done, int(c.stem.split("_")[1]))
            except Exception:  # noqa: BLE001
                pass
    ser = read_tb(name, max_points=5) if (d / "tb").exists() else {}
    if not ser and not meta:
        continue
    finished = bool(meta.get("wall_seconds"))

    def last3(k):
        pts = [v for _, v in ser.get(k, [])]
        return " ".join(f"{v:8.3f}" for v in pts[-3:]) if pts else "       -"

    head = (f"{name}: plan={meta.get('steps', '?'):>8} done>={done:>7} "
            f"{'完成' if finished else '训练中'}")
    print(head)
    keys = DETAIL if verbose else ["time/fps", "trex/success_rate", "trex/fall_rate",
                                  "rollout/ep_len_mean", "trex/com_sway",
                                  "trex/tail_momentum_share"]
    for k in keys:
        print(f"   {k:26s} {last3(k)}")
    if (d / "metrics.json").exists():
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        print(f"   best_score={m.get('best_score')} episodes={m.get('episodes')}")
