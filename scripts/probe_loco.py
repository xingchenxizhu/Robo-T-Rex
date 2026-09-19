"""阶段三进展监控：对最新检查点做确定性评估（训练期指标已被证明会被噪声污染）。

    .venv\\Scripts\\python.exe scripts\\probe_loco.py [--run runs/s3_loco] [--episodes 12]
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default="runs/s3_loco")
    ap.add_argument("--task", default="loco")
    ap.add_argument("--episodes", type=int, default=12)
    args = ap.parse_args()

    run = ROOT / args.run
    cks = sorted(run.glob("checkpoints/*.zip"),
                 key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    final = run / "final_model.zip"
    target = final if final.exists() else (cks[-1] if cks else None)
    if target is None:
        print("还没有检查点")
        return 1
    print(f"确定性评估 {target.relative_to(ROOT)}")
    out = ROOT / "evaluations" / "_probe_loco.json"
    r = subprocess.run([sys.executable, str(ROOT / "scripts" / "evaluate.py"),
                        "--model", str(target.relative_to(ROOT)), "--task", args.task,
                        "--episodes", str(args.episodes), "--out", str(out.relative_to(ROOT))],
                       cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not out.exists():
        print("评估失败")
        return 1
    s = json.loads(out.read_text(encoding="utf-8"))["summary"]
    keys = ["success", "fallen", "distance", "speed_max", "mean_track_err",
            "flight_ratio", "run_flight_events", "max_airtime", "extra_steps",
            "support_switches", "energy_j", "com_sway"]
    for k in keys:
        v = s.get(k)
        print(f"  {k:20s} = {'—' if v is None else f'{v:.4f}'}")
    ok = (s.get("distance") or 0) > 1.5 and (s.get("mean_track_err") or 9) < 0.45
    print("  ->", "已达成前进+跟踪判据" if ok else "仍未前进")
    out.unlink(missing_ok=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
