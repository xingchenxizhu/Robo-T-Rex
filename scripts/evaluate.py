"""策略评估（单条件）。

示例：
    .venv\\Scripts\\python.exe scripts\\evaluate.py --model runs/s1_stand/best_model --task stand ^
        --episodes 40 --perturb --out evaluations/s1_stand_active.json --trace
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trex import TrexEnv  # noqa: E402
from trex.rollout import run_episodes, write_json  # noqa: E402

SHOW = {
    "stand": ["success", "fallen", "com_sway", "pitch_rms", "energy_j", "tail_momentum_share",
              "peak_torque_nm", "saturation_fraction", "recovery_s"],
    "reach": ["success", "fallen", "reached", "bitten", "stood_again", "tail_rest_ok",
              "return_sway_rad", "return_settle_s", "com_sway", "tail_momentum_share"],
    "loco": ["success", "fallen", "distance", "speed_max", "mean_track_err", "flight_ratio",
             "run_flight_events", "max_airtime", "energy_j", "tail_momentum_share"],
    "chase": ["success", "fallen", "attack_time", "bitten", "prey_dist", "distance",
              "speed_max", "mean_track_err", "mean_yaw_track_err", "yaw_travel_deg",
              "energy_j", "tail_momentum_share"],
}


def resolve(path_str: str) -> Path:
    p = Path(path_str)
    return p if p.is_absolute() else ROOT / p


def load_model(path: Path, env):
    if path.suffix != ".zip" and path.with_suffix(".zip").exists():
        path = path.with_suffix(".zip")
    return PPO.load(str(path), env=env, device="cpu")


def build_env(args):
    env = TrexEnv(task=args.task, mode=args.mode, seed=args.seed,
                  perturb=args.perturb, curriculum=args.curriculum,
                  record_trace=args.trace)
    if args.speed_scale is not None:
        env.speed_scale = args.speed_scale
    if args.target_randomize is not None:
        env.target_randomize = args.target_randomize
    return env


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--task", required=True, choices=["stand", "reach", "loco", "chase"])
    ap.add_argument("--mode", default="active", choices=["active", "passive", "fixed"])
    ap.add_argument("--episodes", type=int, default=30)
    ap.add_argument("--seed", type=int, default=12345)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--curriculum", type=int, default=0)
    ap.add_argument("--speed-scale", type=float, default=None)
    ap.add_argument("--target-randomize", type=float, default=None)
    ap.add_argument("--stochastic", action="store_true")
    ap.add_argument("--trace", action="store_true")
    ap.add_argument("--save-trace", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    env = build_env(args)
    model = load_model(resolve(args.model), env)
    res = run_episodes(
        env,
        lambda obs: model.predict(obs, deterministic=not args.stochastic)[0],
        args.episodes, seed=args.seed, collect_analysis=args.trace,
    )
    s = res["summary"]

    print(f"\n=== 评估: {args.task} / 尾巴模式={args.mode} / {s['episodes']} 回合 ===")
    for k in SHOW[args.task]:
        v = s.get(k)
        print(f"  {k:24s} = {'None' if v is None else f'{v:.4f}'}")
    if "coordination" in s:
        print("  -- 尾巴协同分析（测量值，无预设相位）--")
        for k in ("tail_pitch_lag_s", "tail_pitch_corr", "tail_yaw_lag_s", "tail_yaw_corr",
                  "tail_curv_dominant_hz", "tail_curv_spectral_entropy",
                  "tail_tip_spectral_entropy", "tail_qd_rms", "pitch_std",
                  "speed_track_err_mean", "yaw_track_err_mean", "yaw_travel_deg"):
            print(f"  {k:28s} = {s['coordination'].get(k, 0.0):.4f}")
    print(f"  tail_rom_deg             = {np.round(s['tail_rom_deg'], 1).tolist()}")

    if args.out:
        payload = {
            "task": args.task, "mode": args.mode, "perturb": args.perturb,
            "model": args.model, "seed": args.seed, "summary": s,
            "per_episode": res["rows"],
        }
        if res["analyses"]:
            payload["coordination_per_episode"] = res["analyses"]
        print(f"[written] {write_json(resolve(args.out), payload)}")
    if args.save_trace and res["last_trace"] is not None:
        print(f"[written] {write_json(resolve(args.save_trace), res['last_trace'])}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
