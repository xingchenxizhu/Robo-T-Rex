"""改动后的快速回归检查。"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trex import TrexEnv  # noqa: E402

for task in ("stand", "reach", "loco", "chase"):
    env = TrexEnv(task=task, mode="active", seed=0)
    obs, _ = env.reset(seed=0)
    print(f"  {task:6s} obs={env.observation_space.shape[0]:3d} "
          f"act={env.action_space.shape[0]:3d} dim={env.dimension}")
    if task == "reach":
        print("  reach 目标同步: prey_target", np.round(env.prey_target, 3),
              "mocap", np.round(env.data.mocap_pos[0], 3),
              "bite_site", np.round(env.data.site_xpos[env.bite_site], 3))
    if task == "chase":
        gap0 = float(np.linalg.norm(env.data.site_xpos[env.mouth_site]
                                    - env.data.site_xpos[env.bite_site]))
        print(f"  chase 初始嘴-猎物间距 = {gap0:.3f} m "
              f"(应在 1.4 左右，过小会让接近阶段被架空)")
    rng = np.random.default_rng(0)
    info = {}
    for _ in range(900):
        a = np.clip(rng.normal(0, 0.12, env.action_space.shape), -1, 1).astype(np.float32)
        obs, r, term, trunc, info = env.step(a)
        if term or trunc:
            break
    extra = {k: info.get(k) for k in
             ("success_locomotion", "success_run", "success_approach", "speed_max", "extra_steps")
             if k in info}
    print(f"  {task:6s} steps={env.steps:4d} fallen={info['fallen']} "
          f"success={info.get('success')} {extra}")
    env.close()
print("OK")
