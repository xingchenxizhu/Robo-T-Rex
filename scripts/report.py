"""从训练/评估产物自动生成实验报告的数据部分。

    .venv\\Scripts\\python.exe scripts\\report.py --out REPORT_generated.md

读取：
  runs/*/train_meta.json     训练设置与实际用时
  runs/*/tb/**               训练曲线（TensorBoard 事件）
  runs/*/metrics.json        最优窗口指标
  evaluations/*.json         单条件评估
  evaluations/*/compare.json 尾巴对照
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STAGES = [("stand", "阶段一 站立平衡", "s1_stand"),
          ("reach", "阶段二 前探咬合收回", "s2_reach"),
          ("loco", "阶段三 奔跑减速停止", "s3_loco"),
          ("chase", "阶段四 转向追逐", "s4_chase")]
MODES = ["active", "passive", "fixed"]
MODE_CN = {"active": "主动尾巴", "passive": "被动尾巴", "fixed": "刚性尾巴"}


def load_json(p: Path):
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:  # noqa: BLE001
        return None


def tb_series(run_name: str):
    sys.path.insert(0, str(ROOT))
    from app import read_tb

    return read_tb(run_name, max_points=400)


def last(vals, n=1):
    return vals[-n:] if vals else []


def fmt(v, nd=4):
    if v is None:
        return "—"
    if isinstance(v, float) and (abs(v) < 1e-4 and v != 0):
        return f"{v:.2e}"
    return f"{v:.{nd}f}" if isinstance(v, float) else str(v)


def section_training(lines):
    lines += ["## 1. 训练记录（实际执行）", ""]
    lines += ["| 阶段 | 步数 | 并行环境 | 用时(分) | 步/秒 | 最终窗口成功率 | 最终窗口跌倒率 | 最优成功率 |",
              "|---|---|---|---|---|---|---|---|"]
    for task, cn, run in STAGES:
        meta = load_json(ROOT / "runs" / run / "train_meta.json") or {}
        metrics = load_json(ROOT / "runs" / run / "metrics.json") or {}
        ser = tb_series(run) if (ROOT / "runs" / run / "tb").exists() else {}
        sr = [v for _, v in ser.get("trex/success_rate", [])]
        fr = [v for _, v in ser.get("trex/fall_rate", [])]
        wall = meta.get("wall_seconds")
        sps = meta.get("steps_per_second")
        lines.append(
            f"| {cn} | {meta.get('steps', '—')} | {meta.get('envs', '—')} | "
            f"{('%.1f' % (wall / 60)) if wall else '—'} | "
            f"{('%.0f' % sps) if sps else '—'} | "
            f"{fmt(np.mean(last(sr, 20)), 3) if sr else '—'} | "
            f"{fmt(np.mean(last(fr, 20)), 3) if fr else '—'} | "
            f"{fmt(metrics.get('best_score'), 3)} |"
        )
    lines += ["", "> 训练期的成功率是在**带探索噪声**的采样动作下统计的；",
              "> 下面的评估表用**确定性**动作（取分布均值），两者不可直接比较。", ""]
    return lines


def section_curves(lines):
    lines += ["### 训练曲线", ""]
    for task, cn, run in STAGES:
        if not (ROOT / "runs" / run / "tb").exists():
            continue
        ser = tb_series(run)
        if not ser:
            continue
        keys = ["rollout/ep_rew_mean", "trex/success_rate", "trex/fall_rate",
                "trex/com_sway", "trex/tail_rom_mean", "trex/tail_momentum_share",
                "trex/energy_j", "trex/saturation_fraction"]
        lines += [f"**{cn}**", "", "| 指标 | 起点 | 1/4 | 1/2 | 3/4 | 终点 |",
                  "|---|---|---|---|---|---|"]
        for k in keys:
            pts = [v for _, v in ser.get(k, [])]
            if not pts:
                continue
            n = len(pts)
            idx = [0, n // 4, n // 2, (3 * n) // 4, n - 1]
            cells = [fmt(pts[i], 3) for i in idx]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines.append("")
    return lines


def section_eval(lines):
    lines += ["## 2. 策略评估（确定性动作）", ""]
    keys = ["success", "fallen", "com_sway", "pitch_rms", "roll_rms", "energy_j",
            "peak_torque_nm", "saturation_fraction", "slip_m", "tail_momentum_share",
            "recovery_s", "distance", "speed_max", "mean_track_err",
            "mean_yaw_track_err", "yaw_travel_deg", "flight_ratio",
            "run_flight_events", "max_airtime", "return_sway_rad", "return_settle_s",
            "reached", "bitten", "attack_time", "extra_steps"]
    lines += ["| 指标 | " + " | ".join(cn for _, cn, _ in STAGES) + " |",
              "|---|" + "---|" * len(STAGES)]
    data = {}
    for task, cn, run in STAGES:
        d = load_json(ROOT / "evaluations" / f"{run}_active.json")
        data[task] = (d or {}).get("summary", {})
    for k in keys:
        if not any(k in data[t] for t, _, _ in STAGES):
            continue
        cells = []
        for t, _, _ in STAGES:
            v = data[t].get(k)
            cells.append(fmt(v, 4) if v is not None else "—")
        lines.append(f"| {k} | " + " | ".join(cells) + " |")
    lines += ["", "尾关节行程（deg，逐关节）：", ""]
    for task, cn, run in STAGES:
        rom = data[task].get("tail_rom_deg")
        if rom:
            lines.append(f"- {cn}: {np.round(rom, 1).tolist()}")
    lines.append("")
    return lines, data


def section_compare(lines, data):
    lines += ["## 3. 尾巴对照实验（同一策略 × 三种尾巴驱动方式）", "",
              "对照定义：三者的 XML 形态、质量、惯量、关节范围、被动阻尼**逐字相同**，",
              "只改变尾关节是否受执行器驱动。`passive` / `fixed` **没有单独训练**，",
              "本表是固定策略做驱动方式消融的结果。", ""]
    for task, cn, run in STAGES:
        d = load_json(ROOT / "evaluations" / run / "compare.json")
        if not d:
            continue
        s = d["summary"]
        lines += [f"### {cn}", "",
                  "| 指标 | 主动尾巴 | 被动尾巴 | 刚性尾巴 |", "|---|---|---|---|"]
        keys = ["success", "fallen", "com_sway", "pitch_rms", "roll_rms", "energy_j",
                "peak_torque_nm", "saturation_fraction", "distance", "mean_track_err",
                "flight_ratio", "run_flight_events", "recovery_s", "return_sway_rad",
                "extra_steps", "tail_momentum_share"]
        for k in keys:
            if not any(k in s[m] for m in MODES):
                continue
            cells = [fmt(s[m].get(k), 4) if s[m].get(k) is not None else "—" for m in MODES]
            lines.append(f"| {k} | " + " | ".join(cells) + " |")
        coord_keys = ["tail_pitch_lag_s", "tail_pitch_corr", "tail_yaw_lag_s", "tail_yaw_corr",
                      "tail_curv_dominant_hz", "tail_curv_spectral_entropy",
                      "tail_curv_spectral_entropy", "tail_qd_rms",
                      "speed_track_err_mean", "yaw_track_err_mean", "yaw_travel_deg",
                      "yaw_range_deg"]
        cc = {m: s[m].get("coordination") or {} for m in MODES}
        if any(cc[m] for m in MODES):
            lines += ["", "尾巴-身体协同（测量值，无预设相位）：", "",
                      "| 指标 | 主动尾巴 | 被动尾巴 | 刚性尾巴 |", "|---|---|---|---|"]
            for k in coord_keys:
                cells = [fmt(cc[m].get(k), 4) if cc[m].get(k) is not None else "—" for m in MODES]
                lines.append(f"| {k} | " + " | ".join(cells) + " |")
        lines += ["", f"尾关节行程（deg）："]
        for m in MODES:
            rom = s[m].get("tail_rom_deg")
            if rom:
                lines.append(f"- {MODE_CN[m]}: {np.round(rom, 1).tolist()}")
        lines.append("")
    return lines


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="REPORT_generated.md")
    args = ap.parse_args()

    lines = ["<!-- 本文件由 scripts/report.py 自动生成，数字全部来自实际产物 -->", "",
             "# 实验数据附录（自动生成）", ""]
    lines = section_training(lines)
    lines = section_curves(lines)
    lines, data = section_eval(lines)
    lines = section_compare(lines, data)

    out = ROOT / args.out
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"[written] {out}  ({len(lines)} 行)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
