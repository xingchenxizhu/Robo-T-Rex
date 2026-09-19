"""一次跑完全部四个阶段：顺序训练（热启动）→ 自动评估 → 自动尾巴对照。

    .venv\\Scripts\\python.exe scripts\\run_all.py --scale 1.0

设计说明：
  * 只训练 active 尾巴（阶段一到四）。passive / fixed 不训练，
    评估阶段对同一策略做尾巴驱动方式消融。
  * **每个阶段用独立子进程**（`subprocess` + 继承 stdio）：
    这样代码改动会被后续阶段采纳、内存互不污染、单阶段崩溃也不会拖垮整条链。
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

STAGES = [
    # (task, 基础步数, 评估时是否带扰动)
    ("stand", 800_000, True),
    ("reach", 800_000, False),
    ("loco", 1_200_000, False),
    ("chase", 900_000, False),
]
RUN_NAME = {"stand": "s1_stand", "reach": "s2_reach", "loco": "s3_loco", "chase": "s4_chase"}


def run_script(name: str, argv: list[str]) -> int:
    cmd = [sys.executable, str(ROOT / "scripts" / name)] + argv
    print(f"\n$ {' '.join([Path(cmd[1]).name] + argv)}", flush=True)
    t = time.time()
    r = subprocess.run(cmd, cwd=str(ROOT), stdout=sys.stdout, stderr=sys.stderr)
    print(f"[{name}] returncode={r.returncode} 用时 {(time.time()-t)/60:.1f} 分钟", flush=True)
    return r.returncode


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scale", type=float, default=1.0, help="预算缩放系数")
    ap.add_argument("--envs", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--episodes", type=int, default=30, help="评估回合数")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--skip-eval", action="store_true")
    ap.add_argument("--only", default=None, help="只跑某个阶段 (stand/reach/loco/chase)")
    ap.add_argument("--from", dest="from_stage", default=None,
                    help="从某个阶段开始（前面的阶段复用已有检查点）")
    args = ap.parse_args()

    plan = [s for s in STAGES if args.only is None or s[0] == args.only]
    if args.from_stage:
        names = [s[0] for s in STAGES]
        if args.from_stage not in names:
            raise SystemExit(f"--from 必须是 {names} 之一")
        plan = [s for s in STAGES if names.index(s[0]) >= names.index(args.from_stage)]
    summary = {"started": time.strftime("%Y-%m-%d %H:%M:%S"), "stages": {}}
    t0 = time.time()

    prev_ckpt = None
    if plan and plan[0][0] != "stand" and not args.skip_train:
        # 续跑时用上一阶段已有的检查点热启动（同样优先 final_model）
        idx = [s[0] for s in STAGES].index(plan[0][0])
        prev_run = RUN_NAME[STAGES[idx - 1][0]]
        for fname in ("final_model.zip", "best_model.zip"):
            cand = ROOT / "runs" / prev_run / fname
            if cand.exists():
                prev_ckpt = str(cand.relative_to(ROOT))
                print(f"[warm-start] 从 {prev_ckpt} 热启动", flush=True)
                break

    for task, base_steps, perturb_eval in plan:
        name = RUN_NAME[task]
        steps = int(base_steps * args.scale)
        outdir = ROOT / "runs" / name
        print("\n" + "#" * 78)
        print(f"# 阶段 {task}: {steps} 步 -> runs/{name}"
              + (f"  (热启动自 {prev_ckpt})" if prev_ckpt else ""))
        print("#" * 78, flush=True)

        if not args.skip_train:
            argv = ["--task", task, "--steps", str(steps), "--envs", str(args.envs),
                    "--out", f"runs/{name}", "--seed", str(args.seed)]
            if prev_ckpt:
                argv += ["--init-from", prev_ckpt]
            t = time.time()
            rc = run_script("train.py", argv)
            summary["stages"].setdefault(task, {}).update(
                {"train_returncode": rc, "train_seconds": time.time() - t, "steps": steps})

        best = outdir / "best_model.zip"
        final = outdir / "final_model.zip"
        # 主结果用 final_model：课程会让"训练期成功率"在不同难度下不可比，
        # best_model 可能停在难度较低（例如扰动尚未开启）的阶段。
        ckpt = final if final.exists() else (best if best.exists() else None)

        if ckpt is not None and not args.skip_eval:
            rel = str(ckpt.relative_to(ROOT))
            t = time.time()
            ev = ["--model", rel, "--task", task, "--episodes", str(args.episodes), "--trace",
                  "--out", f"evaluations/{name}_active.json"]
            if perturb_eval:
                ev.append("--perturb")
            run_script("evaluate.py", ev)

            cmp_argv = ["--model", rel, "--task", task, "--episodes", str(args.episodes),
                        "--trace", "--outdir", f"evaluations/{name}"]
            if perturb_eval:
                cmp_argv.append("--perturb")
            run_script("compare.py", cmp_argv)

            # 纯物理对照（不加载策略），看形态本身的差异
            ph_argv = ["--zero-policy", "--task", task, "--episodes",
                       str(max(10, args.episodes // 2)), "--outdir", f"evaluations/{name}_physics"]
            if perturb_eval:
                ph_argv.append("--perturb")
            run_script("compare.py", ph_argv)
            summary["stages"].setdefault(task, {})["eval_seconds"] = time.time() - t

        if ckpt is not None:
            prev_ckpt = str(ckpt.relative_to(ROOT))
        else:
            print(f"[warn] 阶段 {task} 没有产出检查点，下一阶段将从头训练")
            prev_ckpt = None

    summary["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    summary["wall_seconds"] = time.time() - t0
    (ROOT / "runs" / "run_all_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n全部完成，总用时 {summary['wall_seconds']/3600:.2f} 小时")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
