"""策略滚动评估（evaluate.py 与 compare.py 共用）。"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .analysis import aggregate, analyse_trace

SCALAR_KEYS = [
    "success", "fallen", "reached", "bitten", "stood_again", "tail_rest_ok",
    "com_sway", "pitch_rms", "roll_rms", "energy_j", "saturation_fraction",
    "peak_torque_nm", "slip_m", "tail_momentum_share", "tail_momentum_abs",
    "total_momentum_abs", "mean_track_err", "mean_yaw_track_err", "yaw_travel_deg",
    "distance", "speed_max", "flight_ratio",
    "run_flight_events", "support_switches", "extra_steps", "max_airtime",
    "duty_left", "duty_right", "prey_dist", "mouth_to_target", "mouth_to_surface",
    "jaw_force", "prey_force", "prey_contact",
]
OPTIONAL_KEYS = ["recovery_s", "return_sway_rad", "return_settle_s", "attack_time"]


def run_episodes(env, predict, n_episodes: int, seed: int = 0, collect_analysis: bool = False):
    rows, analyses, roms = [], [], []
    last_trace = None
    for ep in range(n_episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        info = {}
        while not done:
            obs, _, term, trunc, info = env.step(predict(obs))
            done = term or trunc
        row = {"episode": ep, "steps": env.steps}
        for k in SCALAR_KEYS:
            v = info.get(k)
            row[k] = float(v) if isinstance(v, (bool, int, float)) else 0.0
        for k in OPTIONAL_KEYS:
            v = info.get(k)
            row[k] = None if v is None else float(v)
        rows.append(row)
        roms.append(info.get("tail_rom_deg") or [0.0])
        if collect_analysis and env.trace is not None:
            analyses.append(analyse_trace(env.trace, env.dt))
            last_trace = env.trace

    summary: dict = {}
    for k in SCALAR_KEYS:
        vals = [r[k] for r in rows if isinstance(r[k], (int, float))]
        summary[k] = float(np.mean(vals)) if vals else 0.0
    for k in OPTIONAL_KEYS:
        vals = [r[k] for r in rows if r[k] is not None]
        summary[k] = float(np.mean(vals)) if vals else None
    summary["episodes"] = len(rows)
    summary["tail_rom_deg"] = np.mean(np.array(roms, dtype=float), axis=0).tolist()
    if analyses:
        summary["coordination"] = aggregate(analyses)
    return {"summary": summary, "rows": rows, "analyses": analyses, "last_trace": last_trace}


def write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    import json

    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2, default=float), encoding="utf-8")
    return path
