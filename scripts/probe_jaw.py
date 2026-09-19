"""检查下颌执行器的实际权限：给恒定闭颌指令，看关节角能不能真的闭合。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trex import TrexEnv  # noqa: E402

env = TrexEnv(task="reach", mode="active", seed=0)
env.reset(seed=0)
ji = env.jaw_idx
print("jaw_idx =", ji, " name =", env.names[ji])
print("jnt_range =", np.round(np.degrees(env.model.jnt_range[env.jids[ji]]), 2), "deg")
print("q_lo/q_hi =", np.round(np.degrees([env.q_lo[ji], env.q_hi[ji]]), 2), "deg")
print("home =", round(env.home[ji], 4), " action_scale =", env.action_scale[ji])
print("kp =", env.kp[ji], " kd =", env.kd[ji],
      " torque_limit =", env.torque_limits[ji], " speed_limit =", env.speed_limits[ji])
print()
for jaw_a in (-1.0, -0.5, -0.3, -0.2, -0.15):
    env.reset(seed=0)
    a = np.zeros(env.action_space.shape, np.float32)
    a[ji] = jaw_a
    qs = []
    for _ in range(150):
        env.step(a)
        qs.append(float(env.data.qpos[env.qadr[ji]]))
    tgt = env.home[ji] + jaw_a * env.action_scale[ji]
    print(f"jaw action={jaw_a:+.2f} -> 指令目标 {tgt:.3f} rad ({np.degrees(tgt):5.1f} deg) | "
          f"实际 末 {qs[-1]:.3f} ({np.degrees(qs[-1]):5.1f} deg)  最小 {np.min(qs):.3f} "
          f"({np.degrees(np.min(qs)):5.1f} deg)")

print("\n单步响应（前一步闭颌，随后回零动作）:")
for jaw_a in (-0.6, -0.4, -0.3):
    env.reset(seed=0)
    a = np.zeros(env.action_space.shape, np.float32)
    a[ji] = jaw_a
    env.step(a)
    q1 = float(env.data.qpos[env.qadr[ji]])
    a[ji] = 0.0
    for _ in range(3):
        env.step(a)
    q4 = float(env.data.qpos[env.qadr[ji]])
    print(f"  action={jaw_a:+.2f} -> 1 步后 {q1:.3f} rad ({np.degrees(q1):5.1f} deg)"
          f"  再过 3 步 {q4:.3f} ({np.degrees(q4):5.1f} deg)")
env.close()
