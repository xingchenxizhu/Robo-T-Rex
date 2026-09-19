"""环境工厂（供 DummyVecEnv 使用；本机沙箱禁用多进程管道，故不用 SubprocVecEnv）。"""

from __future__ import annotations

from stable_baselines3.common.monitor import Monitor

from .env import TrexEnv


def make_env(
    task: str,
    mode: str = "active",
    seed: int = 0,
    rank: int = 0,
    perturb: bool = False,
    curriculum: int = 0,
    record_trace: bool = False,
):
    """返回一个 thunk，供 DummyVecEnv 调用。"""

    def _init():
        env = TrexEnv(
            task=task,
            mode=mode,
            seed=seed + rank,
            perturb=perturb,
            curriculum=curriculum,
            record_trace=record_trace,
        )
        env.reset(seed=seed + rank)
        return Monitor(env)

    return _init
