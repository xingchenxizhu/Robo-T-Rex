"""开环步态探测：用简谐的髋/膝/踝指令驱动 3D 环境，看机器人物理上能否产生净前进。

回答的问题：阶段三"原地不动"是
  (a) 环境/几何/摩擦导致的物理上无法前进，还是
  (b) 只是策略没有探索到。

同时顺带标定：腿部动作尺度 0.45 rad 是否够支撑一个步幅。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trex import TrexEnv  # noqa: E402


def idx(env, name):
    return env.names.index(name)


def trial(amp, freq, knee_amp, ankle_gain, duration=11.5, seed=0):
    env = TrexEnv(task="loco", mode="active", seed=seed)
    env.reset(seed=seed)
    a = np.zeros(env.action_space.shape, np.float32)
    names = {n: i for i, n in enumerate(env.names)}
    # 去掉阶段课程对速度的影响：直接用物理推进，不看奖励
    x0 = float(env.data.xpos[env.torso_id, 0])
    t = 0.0
    while t < duration:
        ph = 2 * np.pi * freq * t
        for side, off in (("left", 0.0), ("right", np.pi)):
            s = np.sin(ph + off)
            swing = max(0.0, np.sin(ph + off - np.pi / 2))  # 摆动相
            if f"{side}_hip" in names:
                a[names[f"{side}_hip"]] = amp * s
            if f"{side}_knee" in names:
                a[names[f"{side}_knee"]] = -knee_amp * swing
            if f"{side}_ankle" in names:
                a[names[f"{side}_ankle"]] = -ankle_gain * amp * s
        a = np.clip(a, -1, 1).astype(np.float32)
        env.step(a)
        t = env.elapsed
        if env.fallen:
            break
    x1 = float(env.data.xpos[env.torso_id, 0])
    res = dict(dx=x1 - x0, steps=env.steps, fallen=bool(env.fallen),
               height=float(env.data.xpos[env.torso_id, 2]))
    env.close()
    return res


print("开环简谐步态扫描（3D 环境，无视奖励，只看能不能走）")
print(f"{'amp':>5} {'freq':>5} {'knee':>5} {'ankle':>6} | {'净位移(m)':>9} {'步数':>5} {'跌倒':>5}")
best = None
for amp in (0.5, 0.8, 1.0):
    for freq in (0.8, 1.2, 1.6):
        r = trial(amp, freq, 0.6, 0.5)
        print(f"{amp:5.2f} {freq:5.2f} {0.6:5.2f} {0.5:6.2f} | "
              f"{r['dx']:9.3f} {r['steps']:5d} {str(r['fallen']):>5}")
        if best is None or r["dx"] > best[0]:
            best = (r["dx"], amp, freq)
print(f"\n最好: 位移 {best[0]:.3f} m  (amp={best[1]}, freq={best[2]})")

print("\n结论：" + ("物理上前进是可行的 → 阶段三的问题在探索/信用分配"
                  if best[0] > 0.5 else
                  "开环步态也走不动 → 需要检查动作尺度/几何/摩擦"))
