"""打印四阶段的最终结果表（读取 select_ckpt.py 产出的独立种子复测结果）。

    .venv\\Scripts\\python.exe scripts\\summary_all.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

TASKS = ("stand", "reach", "loco", "chase")
CN = {"stand": "阶段一 站立", "reach": "阶段二 前探咬合", "loco": "阶段三 奔跑", "chase": "阶段四 追逐"}
KEYS = ["success", "fallen", "distance", "speed_max", "mean_track_err",
        "flight_ratio", "run_flight_events", "max_airtime", "duty_left", "duty_right",
        "reached", "bitten", "stood_again", "tail_rest_ok", "return_sway_rad",
        "recovery_s", "com_sway", "extra_steps", "energy_j", "peak_torque_nm",
        "saturation_fraction", "tail_momentum_share"]


def main():
    rows = {}
    for t in TASKS:
        p = ROOT / "evaluations" / f"ckpt_selection_{t}.json"
        if p.exists():
            rows[t] = json.loads(p.read_text(encoding="utf-8"))

    width = 13
    print("%-22s" % "指标" + "".join("%*s" % (width, CN[t].split(" ")[0] + t[:2]) for t in TASKS))
    for k in KEYS:
        cells = []
        for t in TASKS:
            v = (rows.get(t, {}).get("test_summary") or {}).get(k)
            cells.append("—" if v is None else f"{v:.3f}")
        print("%-22s" % k + "".join("%*s" % (width, c) for c in cells))
    print()
    for t in TASKS:
        if t in rows:
            print(f"{CN[t]}: 选中 {rows[t]['chosen']}  "
                  f"选择种子={rows[t]['select_seed']} 复测种子={rows[t]['test_seed']}")
    print("\n注：com_sway 在位移型任务（奔跑/追逐）里等于行进距离量级，不代表晃动；"
          "这类任务看 pitch_rms / mean_track_err。")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
