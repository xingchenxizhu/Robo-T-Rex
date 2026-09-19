"""尾巴-身体协同分析。

回答两个问题：
  1) 尾巴有没有参与全身运动？（角动量占比、与身体运动的耦合强度）
  2) 尾巴是不是"机械摆"？（峰值滞后是否随任务/速度变化；尾部运动的频谱是否单一）

注意：这里所有量都是**测量**出来的，没有任何预设相位或固定延迟。
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class LagResult:
    lags: np.ndarray
    corr: np.ndarray
    peak_lag_s: float
    peak_corr: float
    peak_lag_index: int


def cross_correlation_lag(x, y, dt: float, max_lag_s: float = 0.6) -> LagResult:
    """计算 x 相对 y 的互相关随滞后（秒）的变化。

    正滞后 = x 滞后于 y（x 在 y 之后发生）。
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    x, y = x[:n], y[:n]
    if n < 5 or np.std(x) < 1e-9 or np.std(y) < 1e-9:
        return LagResult(np.zeros(1), np.zeros(1), 0.0, 0.0, 0)
    x = (x - x.mean()) / (x.std() + 1e-12)
    y = (y - y.mean()) / (y.std() + 1e-12)
    max_lag = max(1, int(round(max_lag_s / dt)))
    lags = np.arange(-max_lag, max_lag + 1)
    corr = np.empty(len(lags))
    for k, lag in enumerate(lags):
        if lag >= 0:
            a, b = x[lag:], y[: n - lag] if lag > 0 else y
        else:
            a, b = x[: n + lag], y[-lag:]
        m = min(len(a), len(b))
        corr[k] = float(np.mean(a[:m] * b[:m])) if m > 2 else 0.0
    i = int(np.argmax(np.abs(corr)))
    return LagResult(lags * dt, corr, float(lags[i] * dt), float(corr[i]), i)


def spectrum(x, dt: float):
    """返回 (频率, 幅值, 主频, 谱熵)。谱熵越高说明运动越"丰富"而不是单一机械振荡。"""
    x = np.asarray(x, dtype=float)
    if len(x) < 8:
        return np.zeros(1), np.zeros(1), 0.0, 0.0
    x = x - x.mean()
    win = np.hanning(len(x))
    spec = np.abs(np.fft.rfft(x * win)) ** 2
    freqs = np.fft.rfftfreq(len(x), dt)
    if spec.sum() <= 0:
        return freqs, spec, 0.0, 0.0
    p = spec / spec.sum()
    entropy = float(-np.sum(p[p > 0] * np.log(p[p > 0])) / np.log(len(p))) if len(p) > 1 else 0.0
    return freqs, spec, float(freqs[int(np.argmax(spec))]), entropy


def analyse_trace(trace: dict, dt: float, tail_pitch_dims=None) -> dict:
    """对一条 episode 轨迹做协同分析。"""
    out: dict = {}
    tail_q = np.asarray(trace["tail_q"], dtype=float)          # (T, n_tail_joints)
    if tail_q.ndim == 1:
        tail_q = tail_q[:, None]
    n_tail = tail_q.shape[1]
    # 只取 pitch 关节做矢状面分析（3D 时前一半是 pitch）
    half = n_tail // 2
    pitch_cols = list(range(half)) if half > 0 else list(range(n_tail))
    tail_pitch = tail_q[:, pitch_cols]
    curvature = tail_pitch.sum(axis=1)                          # 尾巴整体弯曲
    curvature_rate = np.gradient(curvature, dt)

    pitch = np.asarray(trace["pitch"], dtype=float)
    roll = np.asarray(trace["roll"], dtype=float)
    yaw_rate = np.asarray(trace["yaw_rate"], dtype=float)
    speed = np.asarray(trace["speed"], dtype=float)
    pitch_rate = np.gradient(pitch, dt)

    # 1) 尾巴弯曲 vs 身体俯仰角速度：滞后是"涌现"的，不是设定的
    lr = cross_correlation_lag(curvature_rate, pitch_rate, dt)
    out["tail_pitch_lag_s"] = lr.peak_lag_s
    out["tail_pitch_corr"] = lr.peak_corr
    out["tail_pitch_lag_curve"] = lr.corr.tolist()
    out["tail_pitch_lag_axis"] = lr.lags.tolist()

    # 2) 尾巴 vs 偏航角速度（横向参与）
    lr_yaw = cross_correlation_lag(curvature_rate, yaw_rate, dt)
    out["tail_yaw_lag_s"] = lr_yaw.peak_lag_s
    out["tail_yaw_corr"] = lr_yaw.peak_corr

    # 3) 尾部运动丰富度
    _, _, f_curv, ent_curv = spectrum(curvature_rate, dt)
    _, _, f_tip, ent_tip = spectrum(tail_pitch[:, -1], dt) if tail_pitch.shape[1] else (None, None, 0.0, 0.0)
    out["tail_curv_dominant_hz"] = float(f_curv)
    out["tail_curv_spectral_entropy"] = float(ent_curv)
    out["tail_tip_dominant_hz"] = float(f_tip)
    out["tail_tip_spectral_entropy"] = float(ent_tip)

    # 4) 行程与速度
    out["tail_rom_deg"] = np.degrees(np.ptp(tail_pitch, axis=0)).tolist()
    out["tail_qd_rms"] = float(np.sqrt(np.mean(np.square(np.gradient(tail_pitch, dt, axis=0)))))

    # 5) 身体晃动
    out["pitch_std"] = float(np.std(pitch))
    out["roll_std"] = float(np.std(roll))
    com_x = np.asarray(trace["com_x"], dtype=float)
    com_y = np.asarray(trace["com_y"], dtype=float)
    out["com_sway"] = float(np.max(np.hypot(com_x - com_x[0], com_y - com_y[0])))
    out["speed_mean"] = float(np.mean(speed))
    out["speed_max"] = float(np.max(np.abs(speed)))

    # 6) 指令跟踪与转向表现
    cmd_speed = np.asarray(trace.get("cmd_speed", np.zeros_like(speed)), dtype=float)
    cmd_yaw = np.asarray(trace.get("cmd_yaw_rate", np.zeros_like(yaw_rate)), dtype=float)
    out["speed_track_err_mean"] = float(np.mean(np.abs(speed - cmd_speed)))
    out["yaw_track_err_mean"] = float(np.mean(np.abs(yaw_rate - cmd_yaw)))
    yaw = np.asarray(trace.get("yaw", np.zeros_like(yaw_rate)), dtype=float)
    if len(yaw) > 1:
        d_yaw = np.diff(yaw)
        d_yaw = np.arctan2(np.sin(d_yaw), np.cos(d_yaw))
        out["yaw_travel_deg"] = float(np.degrees(np.sum(np.abs(d_yaw))))
        out["yaw_net_deg"] = float(np.degrees(yaw[-1] - yaw[0]))
        out["yaw_range_deg"] = float(np.degrees(np.ptp(yaw)))
    else:
        out["yaw_travel_deg"] = 0.0
        out["yaw_net_deg"] = 0.0
        out["yaw_range_deg"] = 0.0
    return out


def aggregate(list_of_dicts, keys=None) -> dict:
    """对多个 episode 的分析结果取均值（跳过 None）。"""
    if not list_of_dicts:
        return {}
    keys = keys or [
        k for k, v in list_of_dicts[0].items()
        if isinstance(v, (int, float)) and v is not None
    ]
    out = {}
    for k in keys:
        vals = [d[k] for d in list_of_dicts if isinstance(d.get(k), (int, float))]
        out[k] = float(np.mean(vals)) if vals else 0.0
    return out
