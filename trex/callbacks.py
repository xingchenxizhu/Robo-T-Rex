"""训练回调：课程调度 + 指标统计 / 最优模型保存。"""

from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback


def unwrap(env):
    """剥掉 Monitor / TimeLimit 等包装，拿到 TrexEnv。"""
    while hasattr(env, "env") and not hasattr(env, "task"):
        env = env.env
    return env


class CurriculumCallback(BaseCallback):
    """按训练进度比例，在线调整底层 TrexEnv 的课程参数。"""

    def __init__(self, schedule, total_steps: int, verbose: int = 0):
        super().__init__(verbose)
        self.schedule = sorted(schedule, key=lambda x: x[0])
        self.total_steps = max(1, int(total_steps))
        self.applied = set()

    def _on_step(self) -> bool:
        frac = self.num_timesteps / self.total_steps
        for f, kwargs in self.schedule:
            if f not in self.applied and frac >= f:
                self.applied.add(f)
                envs = getattr(self.training_env, "envs", [self.training_env])
                for e in envs:
                    target = unwrap(e)
                    for k, v in kwargs.items():
                        setattr(target, k, v)
                if self.verbose:
                    print(f"[curriculum] {frac*100:5.1f}% -> {kwargs}")
        return True


def unwrap_env(env):
    return unwrap(env)


METRIC_KEYS = (
    "com_sway", "pitch_rms", "roll_rms", "energy_j", "saturation_fraction",
    "peak_torque_nm", "slip_m", "tail_momentum_share", "tail_momentum_abs",
    "total_momentum_abs", "return_sway_rad", "recovery_s", "mean_track_err",
    "distance", "flight_ratio", "run_flight_events", "support_switches",
    "extra_steps", "max_airtime", "attack_time",
)


class MetricsCallback(BaseCallback):
    """从 episode 结束的 info 里收集指标，写 TensorBoard，并在改善时保存最优模型。"""

    def __init__(self, log_dir, window: int = 60, save_best: bool = True,
                 best_key="success_rate", print_every: int = 20, verbose: int = 0):
        super().__init__(verbose)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.window = window
        self.save_best = save_best
        self.best_key = best_key
        self.print_every = print_every
        self.records: list[dict] = []
        self.window_records = deque(maxlen=window)
        self.episodes = 0
        self.best = -np.inf
        self.history = defaultdict(list)
        self.best_path = self.log_dir / "best_model.zip"
        self._last_logged = -1

    # ------------------------------------------------------------------ #
    def _on_step(self) -> bool:
        infos = self.locals.get("infos", [])
        for info in infos:
            if not info.get("episode_end"):
                continue
            rec = self._flatten(info)
            self.records.append(rec)
            self.window_records.append(rec)
            self.episodes += 1
        if (self.episodes and self.episodes % self.print_every == 0
                and self.episodes != self._last_logged
                and len(self.window_records) == self.window):
            self._last_logged = self.episodes
            self._log()
        return True

    def _flatten(self, info: dict) -> dict:
        rec = {
            "success": float(bool(info.get("success", False))),
            "fallen": float(bool(info.get("fallen", False))),
            "gait": info.get("gait", "?"),
            "tail_rom_mean": float(np.mean(info.get("tail_rom_deg", [0.0]))),
            "tail_rom_max": float(np.max(info.get("tail_rom_deg", [0.0]))),
            "reached": float(bool(info.get("reached", False))),
            "bitten": float(bool(info.get("bitten", False))),
            "recovery_s": info.get("recovery_s") if info.get("recovery_s") is not None else -1.0,
            "return_sway_rad": info.get("return_sway_rad") if info.get("return_sway_rad") is not None else -1.0,
            "attack_time": info.get("attack_time") if info.get("attack_time") is not None else -1.0,
        }
        for k in ("success_locomotion", "success_run", "success_approach"):
            if k in info:
                rec[k] = float(bool(info[k]))
        for k in METRIC_KEYS:
            v = info.get(k)
            rec[k] = float(v) if v is not None else 0.0
        terms = info.get("reward_terms", {}) or {}
        for k, v in terms.items():
            rec[f"rew_{k}"] = float(v)
        return rec

    # ------------------------------------------------------------------ #
    def _aggregate(self):
        recs = list(self.window_records)
        agg = {}
        keys = {"success", "fallen", "reached", "bitten"}
        for k in keys:
            agg[k] = float(np.mean([r[k] for r in recs]))
        for k in METRIC_KEYS:
            vals = [r[k] for r in recs if not (k in ("recovery_s", "return_sway_rad", "attack_time") and r[k] < 0)]
            agg[k] = float(np.mean(vals)) if vals else 0.0
        agg["success_rate"] = agg["success"]
        agg["fall_rate"] = agg["fallen"]
        gaits = [r["gait"] for r in recs]
        agg["gait_run_frac"] = float(np.mean([g == "run" for g in gaits]))
        agg["gait_walk_frac"] = float(np.mean([g == "walk" for g in gaits]))
        agg["gait_standing_frac"] = float(np.mean([g == "standing" for g in gaits]))
        agg["tail_rom_mean"] = float(np.mean([r["tail_rom_mean"] for r in recs]))
        sway_vals = [r["return_sway_rad"] for r in recs if r["return_sway_rad"] >= 0]
        agg["return_sway_rad"] = float(np.mean(sway_vals)) if sway_vals else 0.0
        for k in ("success_locomotion", "success_run", "success_approach"):
            vals = [r[k] for r in recs if k in r]
            agg[k] = float(np.mean(vals)) if vals else 0.0
        return agg

    def _score(self, agg) -> float:
        """最优模型判据：可以是单个指标，也可以是加权组合。"""
        if isinstance(self.best_key, str):
            return float(agg.get(self.best_key, 0.0))
        return float(sum(w * agg.get(k, 0.0) for k, w in self.best_key))

    def _log(self):
        if self.logger is None:
            return
        agg = self._aggregate()
        for k, v in agg.items():
            self.logger.record(f"trex/{k}", v)
        self.logger.record("trex/episodes", self.episodes)
        for k, v in agg.items():
            self.history[k].append(float(v))
        if self.save_best:
            score = self._score(agg)
            if score > self.best + 1e-9 and self.episodes >= self.window:
                self.best = score
                self.model.save(str(self.best_path))
                self.logger.record("trex/best_score", self.best)
        if self.verbose:
            line = " ".join(
                f"{k}={agg[k]:.3f}" for k in
                ("success_rate", "fall_rate", "com_sway", "pitch_rms", "tail_rom_mean",
                 "tail_momentum_share", "energy_j")
            )
            print(f"[ep {self.episodes:5d}] {line}")

    # ------------------------------------------------------------------ #
    def dump(self, path):
        """把窗口聚合结果与历史写入 json，供报告/UI 使用。"""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "episodes": self.episodes,
            "best_score": None if self.best == -np.inf else float(self.best),
            "best_key": self.best_key,
            "latest": self._aggregate() if self.window_records else {},
            "history": {k: v for k, v in self.history.items()},
        }
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path
