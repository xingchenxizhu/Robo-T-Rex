# 尾巴对照实验：stand

- 策略来源：`runs\s1_stand\selected_model.zip`（trained-policy）
- 每条件回合数：30，扰动：True，随机种子：777
- 三种条件共享同一形态 / 质量 / 惯量 / 关节范围 / 被动阻尼，只改变尾巴驱动方式。
- `passive` 与 `fixed` **没有单独训练**，是对同一策略的驱动方式消融。

| 指标 | 主动尾巴 (active) | 被动尾巴 (passive) | 刚性尾巴 (fixed) |
|---|---|---|---|
| success | 1.0000 | 0.0000 | 1.0000 |
| fallen | 0.0000 | 0.2000 | 0.0000 |
| com_sway | 0.0818 | 0.2244 | 0.0654 |
| pitch_rms | 0.1225 | 0.2251 | 0.0932 |
| roll_rms | 0.0000 | 0.0000 | 0.0000 |
| energy_j | 26.7956 | 13.9154 | 12.3507 |
| peak_torque_nm | 33.8596 | 39.0840 | 33.7205 |
| saturation_fraction | 0.0000 | 0.0000 | 0.0000 |
| recovery_s | 0.4333 | — | 0.1680 |
| support_switches | 0.0000 | 0.0000 | 0.0000 |
| tail_momentum_share | 1.1248 | 1.4806 | 0.5543 |

## 尾巴协同分析（测量值）

| 指标 | 主动尾巴 (active) | 被动尾巴 (passive) | 刚性尾巴 (fixed) |
|---|---|---|---|
| tail_pitch_lag_s | 0.0940 | -0.2693 | -0.0200 |
| tail_pitch_corr | 0.1665 | -0.3074 | 0.2723 |
| tail_yaw_lag_s | 0.0000 | 0.0000 | 0.0000 |
| tail_yaw_corr | 0.0000 | 0.0000 | 0.0000 |
| tail_curv_dominant_hz | 0.7778 | 0.0000 | 2.6278 |
| tail_curv_spectral_entropy | 0.3947 | 0.1721 | 0.8610 |
| tail_tip_spectral_entropy | 0.3846 | 0.1329 | 0.6597 |
| tail_qd_rms | 0.2389 | 0.7713 | 0.0004 |
| pitch_std | 0.0606 | 0.1035 | 0.0585 |
| roll_std | 0.0000 | 0.0000 | 0.0000 |
| com_sway | 0.0818 | 0.2244 | 0.0654 |
| speed_track_err_mean | 0.0408 | 0.0557 | 0.0363 |
| yaw_track_err_mean | 0.0000 | 0.0000 | 0.0000 |
| yaw_travel_deg | 0.0000 | 0.0000 | 0.0000 |
| yaw_range_deg | 0.0000 | 0.0000 | 0.0000 |

ROI 运动对比（尾关节行程，deg）：

- 主动尾巴 (active): [25.9, 10.9, 7.5, 3.7]
- 被动尾巴 (passive): [46.2, 55.6, 42.6, 46.1]
- 刚性尾巴 (fixed): [0.0, 0.0, 0.0, 0.0]
