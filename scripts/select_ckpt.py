"""检查点选择 + 独立种子复测（避免用测试集挑模型的选择偏差）。

    .venv\\Scripts\\python.exe scripts\\select_ckpt.py --run runs/s3_loco --task loco

流程：
  1) 在所有检查点上用【选择种子集】做确定性评估，挑出最好的一个；
  2) 用【另一组从未用于选择的种子】复测该检查点，报告这组数字。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

KEYS = ["success", "fallen", "distance", "speed_max", "mean_track_err",
        "flight_ratio", "run_flight_events", "max_airtime", "duty_left", "duty_right",
        "support_switches", "energy_j", "com_sway", "peak_torque_nm",
        "saturation_fraction", "tail_momentum_share"]


def evaluate(model: Path, task: str, episodes: int, seed: int, tag: str, perturb: bool = False):
    out = ROOT / "evaluations" / f"_sel_{tag}.json"
    cmd = [sys.executable, str(ROOT / "scripts" / "evaluate.py"),
           "--model", str(model.relative_to(ROOT)), "--task", task,
           "--episodes", str(episodes), "--seed", str(seed),
           "--out", str(out.relative_to(ROOT))]
    if perturb:
        cmd.append("--perturb")
    r = subprocess.run(cmd, cwd=str(ROOT), stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    if r.returncode != 0 or not out.exists():
        return None
    s = json.loads(out.read_text(encoding="utf-8"))["summary"]
    out.unlink(missing_ok=True)
    return s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", required=True)
    ap.add_argument("--task", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--select-seed", type=int, default=12345)
    ap.add_argument("--test-seed", type=int, default=999)
    ap.add_argument("--perturb", action="store_true", help="阶段一需带外力扰动评估")
    args = ap.parse_args()

    run = ROOT / args.run
    cks = sorted(run.glob("checkpoints/*.zip"),
                 key=lambda p: int("".join(c for c in p.stem if c.isdigit()) or 0))
    if (run / "final_model.zip").exists():
        cks.append(run / "final_model.zip")
    if not cks:
        print("没有检查点")
        return 1

    rows = []
    print(f"选择集（seed={args.select_seed}, {args.episodes} 回合, perturb={args.perturb}）")
    print(f"{'检查点':>28} {'success':>8} {'fallen':>7} {'dist':>7} {'track':>7} {'flight':>7}")
    for ck in cks:
        s = evaluate(ck, args.task, args.episodes, args.select_seed, "sel", args.perturb)
        if s is None:
            print(f"{ck.name:>28}  评估失败")
            continue
        rows.append((ck, s))
        print(f"{ck.name:>28} {s['success']:8.2f} {s['fallen']:7.2f} "
              f"{s['distance']:7.3f} {s['mean_track_err']:7.3f} {s['run_flight_events']:7.0f}")

    if not rows:
        return 1
    best_ck, _ = max(rows, key=lambda r: (r[1]["success"], r[1]["distance"]))
    print(f"\n选择集最优: {best_ck.name}")

    print(f"\n复测（独立种子 seed={args.test_seed}, 30 回合）—— 报告用这组数字")
    s = evaluate(best_ck, args.task, 30, args.test_seed, "test", args.perturb)
    for k in KEYS:
        v = s.get(k)
        print(f"  {k:22s} = {'—' if v is None else f'{v:.4f}'}")
    payload = {"chosen": str(best_ck.relative_to(ROOT)), "select_seed": args.select_seed,
               "test_seed": args.test_seed, "test_summary": s,
               "select_table": {c.name: r for c, r in rows}}
    (ROOT / "evaluations" / f"ckpt_selection_{args.task}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    # 把选中的检查点复制成稳定文件名，并把映射记进 selected_checkpoints.json，
    # 供 eval_all.py --ckpt selected 使用。
    import shutil

    stable = run / "selected_model.zip"
    shutil.copyfile(best_ck, stable)
    mapping_path = ROOT / "evaluations" / "selected_checkpoints.json"
    mapping = {}
    if mapping_path.exists():
        try:
            mapping = json.loads(mapping_path.read_text(encoding="utf-8"))
        except Exception:  # noqa: BLE001
            mapping = {}
    mapping[args.task] = str(stable.relative_to(ROOT))
    mapping_path.write_text(json.dumps(mapping, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"[written] {stable.relative_to(ROOT)}  （选中的检查点副本）")
    print("[written] evaluations/ckpt_selection_%s.json, evaluations/selected_checkpoints.json" % args.task)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
