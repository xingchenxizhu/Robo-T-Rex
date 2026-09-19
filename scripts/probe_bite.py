"""诊断阶段二"咬合阶段"的实际状态：接触有没有、闭颌有没有、力多大。

    .venv\\Scripts\\python.exe scripts\\probe_bite.py --model runs/s2_reach/checkpoints/ckpt_240000_steps.zip
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trex import TrexEnv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--episodes", type=int, default=12)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--stochastic", action="store_true")
    args = ap.parse_args()

    env = TrexEnv(task="reach", mode="active", seed=args.seed)
    from stable_baselines3 import PPO

    p = Path(args.model)
    if not p.is_absolute():
        p = ROOT / p
    if p.suffix != ".zip" and p.with_suffix(".zip").exists():
        p = p.with_suffix(".zip")
    model = PPO.load(str(p), env=env, device="cpu")

    jaw_i = env.jaw_idx
    rows = []
    for ep in range(args.episodes):
        obs, _ = env.reset(seed=args.seed + ep)
        done = False
        while not done:
            a, _ = model.predict(obs, deterministic=not args.stochastic)
            obs, r, term, trunc, info = env.step(a)
            done = term or trunc
            # 逐步的 info 是精简版（只有回合结束才有完整 info），
            # 所以这里直接从 env 内部读量。
            ph = env.phase_name()
            mouth = env.data.site_xpos[env.mouth_site]
            surface = max(0.0, float(np.linalg.norm(mouth - env.prey_target)) - env.prey_radius)
            if ph in ("probe", "bite", "retract", "rest"):
                rows.append({
                    "phase": ph,
                    "surface": surface,
                    "contact": float(bool(env._prey_contact)),
                    "force": float(env._prey_force),
                    "jaw": float(env.data.qpos[env.qadr[jaw_i]]),
                    "bitten": float(bool(env.bitten)),
                })
    if not rows:
        print("没有采样到相关阶段")
        return 1
    print(f"采样 {len(rows)} 步（{args.episodes} 回合）")
    for ph in ("probe", "bite", "retract", "rest"):
        sel = [r for r in rows if r["phase"] == ph]
        if not sel:
            continue
        s = np.array([r["surface"] for r in sel])
        c = np.array([r["contact"] for r in sel])
        f = np.array([r["force"] for r in sel])
        j = np.array([r["jaw"] for r in sel])
        print(f"  [{ph:7s}] n={len(sel):4d}  到表面: 最小 {s.min():.3f} 中位 {np.median(s):.3f}"
              f"  接触率 {c.mean():.2f}  力max {f.max():6.2f}  下颌中位 {np.degrees(np.median(j)):5.1f}°")

    bite = [r for r in rows if r["phase"] == "bite"]
    d = {k: np.array([r[k] for r in bite]) for k in bite[0]}
    print(f"\n咬合段（{len(bite)} 步）:")
    print(f"  到表面距离 <0.09 的比例 {np.mean(d['surface'] < 0.09):.2f}，最小 {d['surface'].min():.3f}")
    print(f"  嘴部接触比例 {d['contact'].mean():.2f}；力落在 [0.5,60] 的比例 "
          f"{np.mean((d['force'] >= 0.5) & (d['force'] <= 60)):.2f}")
    print(f"  下颌 <0.16rad({np.degrees(0.16):.1f}°) 的比例 {np.mean(d['jaw'] < 0.16):.2f}")
    ok = (d["surface"] < 0.09) & (d["contact"] > 0) & (d["jaw"] < 0.16) & \
         (d["force"] >= 0.5) & (d["force"] <= 60)
    print(f"  四条件同时满足的步比例: {ok.mean():.3f}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
