"""尾巴对照实验：同一个策略 / 同一套形态，只改变尾巴的驱动方式。

对照定义（重要）：
  active   尾巴关节由策略执行器驱动（训练时使用的模式）
  passive  尾关节执行器力矩恒为 0，尾巴自由被动摆动
  fixed    尾关节用 equality 约束刚性锁死，尾巴是躯干的刚体延伸

三者共享同一份 XML 形态、质量、惯量、关节范围与被动阻尼。
本脚本**不训练** passive/fixed：它做的是"对同一策略做尾巴驱动方式消融"，
即固定策略问"尾巴的执行器权限值多少"。报告里会明确标注这一点。

示例：
    .venv\\Scripts\\python.exe scripts\\compare.py --model runs/s1_stand/best_model ^
        --task stand --episodes 40 --perturb --outdir evaluations/s1_compare
    # 不加载策略，只看纯物理形态差异
    .venv\\Scripts\\python.exe scripts\\compare.py --zero-policy --task stand --episodes 20 ^
        --outdir evaluations/s1_physics
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
# 沙箱下用户目录的 matplotlib 缓存不可写，指到工作区
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplcache"))
(ROOT / ".mplcache").mkdir(exist_ok=True)

import numpy as np  # noqa: E402

from trex import TrexEnv  # noqa: E402
from trex.rollout import run_episodes, write_json  # noqa: E402

MODES = ("active", "passive", "fixed")

COMPARE_KEYS = {
    "stand": ["success", "fallen", "com_sway", "pitch_rms", "roll_rms", "energy_j",
              "peak_torque_nm", "saturation_fraction", "recovery_s", "support_switches",
              "tail_momentum_share"],
    "reach": ["success", "fallen", "reached", "bitten", "stood_again", "tail_rest_ok",
              "return_sway_rad", "return_settle_s", "com_sway", "energy_j",
              "tail_momentum_share", "peak_torque_nm"],
    "loco": ["success", "fallen", "distance", "mean_track_err", "flight_ratio",
             "run_flight_events", "max_airtime", "support_switches", "energy_j",
             "slip_m", "tail_momentum_share"],
    "chase": ["success", "fallen", "attack_time", "bitten", "distance", "prey_dist",
              "mean_track_err", "energy_j", "tail_momentum_share"],
}

LABEL = {"active": "主动尾巴 (active)", "passive": "被动尾巴 (passive)", "fixed": "刚性尾巴 (fixed)"}


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def zero_policy(env):
    return lambda obs: np.zeros(env.action_space.shape, dtype=np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--zero-policy", action="store_true")
    ap.add_argument("--task", required=True, choices=["stand", "reach", "loco", "chase"])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=777)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--curriculum", type=int, default=0)
    ap.add_argument("--speed-scale", type=float, default=None)
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--outdir", required=True)
    args = ap.parse_args()

    if not args.zero_policy and not args.model:
        ap.error("需要 --model 或 --zero-policy")

    model = None
    if args.model:
        from stable_baselines3 import PPO

        probe = TrexEnv(task=args.task, mode="active", seed=args.seed)
        path = resolve(args.model)
        if path.suffix != ".zip" and path.with_suffix(".zip").exists():
            path = path.with_suffix(".zip")
        model = PPO.load(str(path), env=probe, device="cpu")
        probe.close()

    results = {}
    for mode in MODES:
        env = TrexEnv(task=args.task, mode=mode, seed=args.seed,
                      perturb=args.perturb, curriculum=args.curriculum, record_trace=args.trace)
        if args.speed_scale is not None:
            env.speed_scale = args.speed_scale
        if model is None:
            predict = zero_policy(env)
            tag = "zero-policy"
        else:
            predict = lambda obs, _m=model: _m.predict(obs, deterministic=True)[0]
            tag = "trained-policy"
        res = run_episodes(env, predict, args.episodes, seed=args.seed,
                           collect_analysis=args.trace)
        results[mode] = res
        env.close()
        print(f"  [{mode}] success={res['summary']['success']:.3f} "
              f"fallen={res['summary']['fallen']:.3f} "
              f"sway={res['summary']['com_sway']:.3f}")

    keys = COMPARE_KEYS[args.task]
    outdir = resolve(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ---------------- 表格 ----------------
    lines = [
        f"# 尾巴对照实验：{args.task}",
        "",
        f"- 策略来源：`{args.model or '未加载（零动作）'}`（{tag}）",
        f"- 每条件回合数：{args.episodes}，扰动：{args.perturb}，随机种子：{args.seed}",
        "- 三种条件共享同一形态 / 质量 / 惯量 / 关节范围 / 被动阻尼，只改变尾巴驱动方式。",
        "- `passive` 与 `fixed` **没有单独训练**，是对同一策略的驱动方式消融。",
        "",
        "| 指标 | " + " | ".join(LABEL[m] for m in MODES) + " |",
        "|---|" + "---|" * len(MODES),
    ]
    for k in keys:
        cells = []
        for m in MODES:
            v = results[m]["summary"].get(k)
            cells.append("—" if v is None else f"{v:.4f}")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")

    if args.trace:
        lines += ["", "## 尾巴协同分析（测量值）", "",
                  "| 指标 | " + " | ".join(LABEL[m] for m in MODES) + " |",
                  "|---|" + "---|" * len(MODES)]
        for k in ("tail_pitch_lag_s", "tail_pitch_corr", "tail_yaw_lag_s", "tail_yaw_corr",
                  "tail_curv_dominant_hz", "tail_curv_spectral_entropy",
                  "tail_tip_spectral_entropy", "tail_qd_rms", "pitch_std", "roll_std",
                  "com_sway", "speed_track_err_mean", "yaw_track_err_mean",
                  "yaw_travel_deg", "yaw_range_deg"):
            cells = []
            for m in MODES:
                c = results[m]["summary"].get("coordination")
                cells.append(f"{c.get(k, 0.0):.4f}" if c else "—")
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines += ["", "ROI 运动对比（尾关节行程，deg）：", ""]
        for m in MODES:
            rom = np.round(results[m]["summary"]["tail_rom_deg"], 1).tolist()
            lines.append(f"- {LABEL[m]}: {rom}")

    md = outdir / "compare.md"
    md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines[:12]))
    print(f"\n[written] {md}")

    write_json(outdir / "compare.json", {
        "task": args.task, "model": args.model, "zero_policy": args.zero_policy,
        "episodes": args.episodes, "perturb": args.perturb, "seed": args.seed,
        "summary": {m: results[m]["summary"] for m in MODES},
    })
    print(f"[written] {outdir / 'compare.json'}")

    # ---------------- 图 ----------------
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
        plt.rcParams["axes.unicode_minus"] = False

        plot_keys = [k for k in keys if k in
                     ("success", "com_sway", "energy_j", "pitch_rms", "distance",
                      "mean_track_err", "tail_momentum_share", "return_sway_rad",
                      "flight_ratio", "recovery_s")]
        ncol = min(4, len(plot_keys))
        nrow = int(np.ceil(len(plot_keys) / ncol))
        fig, axes = plt.subplots(nrow, ncol, figsize=(4 * ncol, 3 * nrow))
        axes = np.atleast_1d(axes).ravel()
        for ax, k in zip(axes, plot_keys):
            vals = [results[m]["summary"].get(k) or 0.0 for m in MODES]
            ax.bar([LABEL[m].split(" ")[0] for m in MODES], vals,
                   color=["#2e8b57", "#c8860d", "#8b3a3a"])
            ax.set_title(k)
        for ax in axes[len(plot_keys):]:
            ax.axis("off")
        fig.suptitle(f"尾巴对照实验 - {args.task}")
        fig.tight_layout()
        fig.savefig(outdir / "compare.png", dpi=130)
        print(f"[written] {outdir / 'compare.png'}")
    except Exception as exc:  # noqa: BLE001
        print(f"[plot skipped] {type(exc).__name__}: {exc}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
