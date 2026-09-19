"""并行/吞吐基准测试。"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trex import TrexEnv  # noqa: E402


def bench_single(task, n=1500):
    env = TrexEnv(task=task, mode="active", seed=0)
    env.reset(seed=0)
    rng = np.random.default_rng(0)
    a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
    t0 = time.perf_counter()
    steps = 0
    for i in range(n):
        _, _, term, trunc, _ = env.step(a)
        steps += 1
        if term or trunc:
            env.reset(seed=i)
    dt = time.perf_counter() - t0
    print(f"{task:6s} single: {steps} ctrl-steps in {dt:.2f}s -> {steps/dt:8.1f} steps/s "
          f"({env.substeps} substeps/step, {env.model.nv} dof)")


def bench_subproc():
    try:
        from stable_baselines3.common.vec_env import SubprocVecEnv
        from trex.vec import make_env

        n = 4
        venv = SubprocVecEnv([make_env("stand", "active", 1, i) for i in range(n)])
        venv.reset()
        a = np.zeros((n, venv.action_space.shape[0]), np.float32)
        t0 = time.perf_counter()
        for _ in range(200):
            venv.step(a)
        dt = time.perf_counter() - t0
        print(f"SubprocVecEnv({n}) OK: {200*n/dt:8.1f} env-steps/s")
        venv.close()
        return True
    except Exception as exc:  # noqa: BLE001
        print(f"SubprocVecEnv FAIL: {type(exc).__name__}: {exc}")
        return False


if __name__ == "__main__":
    for task in ("stand", "loco"):
        bench_single(task)
    bench_subproc()
