"""对四个阶段统一重跑评估与对照（可选检查点：final / best）。

    .venv\\Scripts\\python.exe scripts\\eval_all.py                 # 用 final_model
    .venv\\Scripts\\python.exe scripts\\eval_all.py --ckpt best_model
    .venv\\Scripts\\python.exe scripts\\eval_all.py --only stand --episodes 40

为什么需要它：课程会让"训练期成功率"在不同难度阶段不可比，
`best_model` 可能停留在难度较低时的检查点。报告的**主结果**一律用
`final_model`（训练结束时的策略）评估；`best_model` 可作对照再跑一次。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGES = [("stand", "s1_stand", True), ("reach", "s2_reach", False),
          ("loco", "s3_loco", False), ("chase", "s4_chase", False)]


def run(name, argv):
    cmd = [sys.executable, str(ROOT / "scripts" / name)] + argv
    print(f"\n$ {name} {' '.join(argv)}", flush=True)
    r = subprocess.run(cmd, cwd=str(ROOT), stdout=sys.stdout, stderr=sys.stderr)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default="final_model",
                    choices=["final_model", "best_model", "selected"],
                    help="selected = 用 select_ckpt.py 按确定性评估选出的检查点")
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--only", default=None)
    ap.add_argument("--suffix", default="active", help="输出文件名后缀，区分不同检查点")
    args = ap.parse_args()

    selected = {}
    sel_path = ROOT / "evaluations" / "selected_checkpoints.json"
    if sel_path.exists():
        try:
            selected = json.loads(sel_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            selected = {}

    summary = {}
    for task, run_name, perturb in STAGES:
        if args.only and task != args.only:
            continue
        if args.ckpt == "selected" and task in selected:
            ck = ROOT / selected[task]
        else:
            ck = ROOT / "runs" / run_name / f"{args.ckpt}.zip"
        if not ck.exists():
            print(f"[skip] {task}: 找不到 {ck}")
            continue
        rel = str(ck.relative_to(ROOT))
        base = [ "--model", rel, "--task", task, "--episodes", str(args.episodes)]
        if perturb:
            base.append("--perturb")

        run("evaluate.py", base + ["--trace",
                                   "--out", f"evaluations/{run_name}_{args.suffix}.json"])
        run("compare.py", base + ["--trace", "--outdir", f"evaluations/{run_name}"])
        run("compare.py", ["--zero-policy", "--task", task,
                           "--episodes", str(max(10, args.episodes // 2)),
                           "--outdir", f"evaluations/{run_name}_physics"]
            + (["--perturb"] if perturb else []))
        ev = ROOT / "evaluations" / f"{run_name}_{args.suffix}.json"
        if ev.exists():
            summary[task] = json.loads(ev.read_text(encoding="utf-8"))["summary"]
    (ROOT / "evaluations" / "eval_all_summary.json").write_text(
        json.dumps({"ckpt": args.ckpt, "episodes": args.episodes, "summary": summary},
                   ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    print("\n[written] evaluations/eval_all_summary.json")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
