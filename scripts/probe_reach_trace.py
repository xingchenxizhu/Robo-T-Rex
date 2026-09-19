"""逐步打印阶段二一回合内 嘴部/目标/下颌 的时间序列，用于定位判据与几何问题。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trex import TrexEnv  # noqa: E402

env = TrexEnv(task="reach", mode="active", seed=3)
obs, _ = env.reset(seed=3)
print("initial_mouth =", np.round(env.initial_mouth, 4))
print("prey_target   =", np.round(env.prey_target, 4))
print("prey_radius   =", env.prey_radius)
print("|target - initial_mouth| =", round(float(np.linalg.norm(env.prey_target - env.initial_mouth)), 4))
print()
print(" step   t   phase    mouth(x,y,z)                |m-tgt|  surface  contact  jaw_deg  reached")
model = None
for i in range(int(env.duration / env.dt)):
    a = np.zeros(env.action_space.shape, np.float32) if model is None else None
    obs, r, term, trunc, info = env.step(a)
    if i % 15 == 0 or env.phase_name() in ("bite",) and i % 5 == 0:
        mouth = env.data.site_xpos[env.mouth_site]
        dist = float(np.linalg.norm(mouth - env.prey_target))
        surf = max(0.0, dist - env.prey_radius)
        jaw = float(np.degrees(env.data.qpos[env.qadr[env.jaw_idx]]))
        print(f" {i:4d} {env.elapsed:5.2f} {env.phase_name():8s} "
              f"{np.round(mouth,3)!s:28s} {dist:7.3f} {surf:8.3f} "
              f"{int(env._prey_contact):8d} {jaw:8.1f} {int(env.reached):8d}")
    if term or trunc:
        break
env.close()
