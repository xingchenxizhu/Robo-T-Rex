"""探测阶段二判据的物理可达性。

1) 嘴部能贴到猎物表面多近（random 动作扫描）
2) 强制闭颌时，能否真的产生"接触 + 接触力在合理区间"的简化咬合

如果第 2 步做不到，说明咬合判据仍然不可达。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trex import TrexEnv, load_config  # noqa: E402

cfg = load_config()
tgt = cfg["target"]
print(f"猎物半径 = {tgt['radius']}  目标距离 = {tgt['start_distance']}")
print(f"reached 阈值(到表面) = {tgt['reach_surface_gap']}  "
      f"bitten 阈值 = {tgt['bite_surface_gap']}  闭颌阈值 = {tgt['jaw_closed']} rad")

best_surface = 1e9
for std in (0.15, 0.3, 0.5):
    mins = []
    for ep in range(6):
        env = TrexEnv(task="reach", mode="active", seed=ep)
        obs, _ = env.reset(seed=ep)
        rng = np.random.default_rng(ep)
        m = 1e9
        for _ in range(int(env.duration / env.dt)):
            a = np.clip(rng.normal(0, std, env.action_space.shape), -1, 1).astype(np.float32)
            obs, r, term, trunc, info = env.step(a)
            m = min(m, float(info.get("mouth_to_surface", 1e9)))
            if term or trunc:
                break
        mins.append(m)
        env.close()
    print(f"  random std={std:.2f}: 最小到表面距离 {np.min(mins):.3f} m")
    best_surface = min(best_surface, float(np.min(mins)))

# 强制闭颌 + 随机其他关节，看简化咬合能否触发
bites = 0
contacts = 0
max_force = 0.0
min_surface_at_contact = 1e9
attempts = 12
for ep in range(attempts):
    env = TrexEnv(task="reach", mode="active", seed=100 + ep)
    env.reset(seed=100 + ep)
    rng = np.random.default_rng(100 + ep)
    jaw = int(np.flatnonzero(env.is_jaw)[0])
    for _ in range(int(env.duration / env.dt)):
        a = np.clip(rng.normal(0, 0.15, env.action_space.shape), -1, 1).astype(np.float32)
        a[jaw] = -1.0            # 一直闭颌
        obs, r, term, trunc, info = env.step(a)
        if info.get("episode_end"):
            if info.get("prey_contact"):
                contacts += 1
                max_force = max(max_force, float(info.get("prey_force", 0.0)))
                min_surface_at_contact = min(min_surface_at_contact,
                                             float(info.get("mouth_to_surface", 1e9)))
            if info.get("bitten"):
                bites += 1
        if term or trunc:
            break
    env.close()

print(f"\n强制闭颌 + 随机动作 {attempts} 回合：")
print(f"  出现过嘴部接触的回合: {contacts}/{attempts}")
print(f"  成功简化为咬合的回合: {bites}/{attempts}")
print(f"  接触时最大接触力: {max_force:.2f} N  (判据要求 "
      f"{tgt['bite_force_min']}~{tgt['bite_force_max']} N)")
print(f"  接触时最小到表面距离: {min_surface_at_contact:.3f} m")
print("\n结论：" + ("咬合判据可达" if bites > 0 else
                  "咬合判据仍不可达 —— 需要继续放宽或改目标几何"))
