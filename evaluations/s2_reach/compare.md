# 尾巴对照实验：reach

- 策略来源：`runs\s2_reach\selected_model.zip`（trained-policy）
- 每条件回合数：30，扰动：False，随机种子：777
- 三种条件共享同一形态 / 质量 / 惯量 / 关节范围 / 被动阻尼，只改变尾巴驱动方式。
- `passive` 与 `fixed` **没有单独训练**，是对同一策略的驱动方式消融。

| 指标 | 主动尾巴 (active) | 被动尾巴 (passive) | 刚性尾巴 (fixed) |
|---|---|---|---|
| success | 1.0000 | 1.0000 | 1.0000 |
| fallen | 0.0000 | 0.0000 | 0.0000 |
| reached | 1.0000 | 1.0000 | 1.0000 |
| bitten | 1.0000 | 1.0000 | 1.0000 |
| stood_again | 1.0000 | 1.0000 | 1.0000 |
| tail_rest_ok | 1.0000 | 1.0000 | 1.0000 |
| return_sway_rad | 0.0063 | 0.0073 | 0.0049 |
| return_settle_s | 0.0000 | 0.0000 | 0.0000 |
| com_sway | 0.0164 | 0.0310 | 0.0059 |
| energy_j | 26.2222 | 11.7382 | 8.7298 |
| tail_momentum_share | 1.4560 | 1.6433 | 0.3230 |
| peak_torque_nm | 33.1298 | 41.3137 | 31.1335 |

## 尾巴协同分析（测量值）

| 指标 | 主动尾巴 (active) | 被动尾巴 (passive) | 刚性尾巴 (fixed) |
|---|---|---|---|
| tail_pitch_lag_s | 0.1200 | 0.1800 | 0.0000 |
| tail_pitch_corr | 0.8093 | 0.4391 | 0.5092 |
| tail_yaw_lag_s | 0.0000 | 0.0000 | 0.0000 |
| tail_yaw_corr | 0.0000 | 0.0000 | 0.0000 |
| tail_curv_dominant_hz | 0.0000 | 0.0000 | 13.5385 |
| tail_curv_spectral_entropy | 0.5205 | 0.1470 | 0.8953 |
| tail_tip_spectral_entropy | 0.4272 | 0.1185 | 0.9375 |
| tail_qd_rms | 0.2548 | 0.7475 | 0.0006 |
| pitch_std | 0.0133 | 0.0104 | 0.0132 |
| roll_std | 0.0000 | 0.0000 | 0.0000 |
| com_sway | 0.0164 | 0.0310 | 0.0059 |
| speed_track_err_mean | 0.0032 | 0.0023 | 0.0023 |
| yaw_track_err_mean | 0.0000 | 0.0000 | 0.0000 |
| yaw_travel_deg | 0.0000 | 0.0000 | 0.0000 |
| yaw_range_deg | 0.0000 | 0.0000 | 0.0000 |

ROI 运动对比（尾关节行程，deg）：

- 主动尾巴 (active): [27.0, 12.2, 8.0, 5.8]
- 被动尾巴 (passive): [46.8, 56.6, 36.3, 45.8]
- 刚性尾巴 (fixed): [0.0, 0.0, 0.0, 0.0]
