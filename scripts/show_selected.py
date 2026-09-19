"""打印每个阶段选中的检查点，以及选择集里各检查点的成绩。

    .venv\\Scripts\\python.exe scripts\\show_selected.py
"""
from __future__ import annotations

import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CN = {"stand": "阶段一 站立", "reach": "阶段二 前探咬合",
      "loco": "阶段三 奔跑", "chase": "阶段四 追逐"}


def main():
    sel_path = ROOT / "evaluations" / "selected_checkpoints.json"
    print("selected_checkpoints.json：")
    if sel_path.exists():
        sel = json.loads(sel_path.read_text(encoding="utf-8"))
        for k, v in sel.items():
            print(f"  {k:6s} -> {v}")

    for t in ("stand", "reach", "loco", "chase"):
        p = ROOT / "evaluations" / f"ckpt_selection_{t}.json"
        if not p.exists():
            continue
        d = json.loads(p.read_text(encoding="utf-8"))
        s = d["test_summary"]
        chosen = Path(d["chosen"]).name
        print(f"\n{CN[t]}：选中 {chosen}   (选择种子 {d['select_seed']} / 复测种子 {d['test_seed']})")
        print("  复测 30 回合：success={:.3f} fallen={:.3f} distance={:.3f} "
              "track={:.3f} flight={:.0f}".format(
                  s.get("success", 0), s.get("fallen", 0), s.get("distance", 0),
                  s.get("mean_track_err", 0), s.get("run_flight_events", 0)))
        tab = d.get("select_table") or {}
        rows = sorted(
            ((n, r) for n, r in tab.items()),
            key=lambda x: (x[1].get("success", 0), x[1].get("distance", 0)),
            reverse=True)
        print("  选择集（seed 12345, 10 回合）按成功率排序：")
        for n, r in rows:
            mark = "  <== 选中" if n == chosen else ""
            print("    {:>30s}  succ={:.2f}  fallen={:.2f}  dist={:6.3f}  track={:.3f}{}".format(
                n, r.get("success", 0), r.get("fallen", 0), r.get("distance", 0),
                r.get("mean_track_err", 0), mark))


if __name__ == "__main__":
    main()
