"""环境自检：模型可运行性 + 尾巴"活性"基线。

运行：
    .venv\\Scripts\\python.exe scripts\\smoke.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trex import TASKS, MODES, TrexEnv, load_config, tail_profile  # noqa: E402


def hr(title):
    print("\n" + "=" * 72)
    print(title)
    print("=" * 72)


def check_models():
    hr("1. 模型构建 / 形态一致性")
    cfg = load_config()
    prof = tail_profile(cfg)
    print(f"尾巴: {prof['segments']} 段, 总长 {prof['length']} m, 总质量 {prof['masses'].sum():.3f} kg")
    print(f"  各段质量   : {np.round(prof['masses'], 4).tolist()}")
    print(f"  各段半径   : {np.round(prof['radii'], 4).tolist()}")
    print(f"  各段力矩上限: {np.round(prof['torques'], 2).tolist()} N·m")
    for task in TASKS:
        env = TrexEnv(task=task, mode="active", seed=0)
        m = env.model
        print(
            f"  {task:6s} dim={env.dimension:7s} nq={m.nq:3d} nv={m.nv:3d} nu={m.nu:3d} "
            f"总质量={m.body_mass.sum():6.3f} kg obs={env.observation_space.shape[0]:3d} "
            f"act={env.action_space.shape[0]:3d} 时长={env.duration:.1f}s"
        )
    # 三种对照的惯量必须一致
    import json

    for dim in ("planar", "3d"):
        vals = []
        for mode in MODES:
            p = ROOT / "models" / f"{dim}_{mode}_inertia.json"
            if not p.exists():
                vals = None
                break
            snap = json.loads(p.read_text(encoding="utf-8"))
            vals.append((snap["total_mass"], snap["body_mass"], snap["nq"], snap["nv"]))
        if vals:
            same = all(
                abs(v[0] - vals[0][0]) < 1e-12 and v[1] == vals[0][1] and v[2] == vals[0][2]
                for v in vals
            )
            print(f"  [{dim}] fixed/passive/active 形态一致: {same}")
    return cfg


def check_runs():
    hr("2. 各任务随机动作滚动（NaN / 崩溃检查）")
    ok = True
    for task in TASKS:
        for mode in MODES:
            env = TrexEnv(task=task, mode=mode, seed=1)
            env.reset(seed=1)
            rng = np.random.default_rng(0)
            total = 0.0
            try:
                for _ in range(int(env.duration / env.dt)):
                    a = rng.uniform(-1, 1, env.action_space.shape).astype(np.float32)
                    _, r, term, trunc, info = env.step(a)
                    total += r
                    if not np.isfinite(r):
                        raise RuntimeError("奖励非有限")
                    if term or trunc:
                        break
            except Exception as exc:  # noqa: BLE001
                ok = False
                print(f"  FAIL {task}/{mode}: {type(exc).__name__}: {exc}")
                continue
            print(
                f"  OK   {task:6s}/{mode:7s} steps={env.steps:4d} reward={total:8.2f} "
                f"fallen={info['fallen']} gait={info['gait']:8s} "
                f"tail_rom={np.round(info['tail_rom_deg'], 1).tolist()}"
            )
    return ok


def check_zero_action():
    hr("3. 零动作站立检查（PD 保持 home 姿态，不摔倒即物理/增益合理）")
    for task in ("stand",):
        for mode in MODES:
            env = TrexEnv(task=task, mode=mode, seed=2)
            env.reset(seed=2)
            zero = np.zeros(env.action_space.shape, np.float32)
            for _ in range(int(env.duration / env.dt)):
                _, _, term, trunc, info = env.step(zero)
                if term or trunc:
                    break
            print(
                f"  {task}/{mode:7s}: 末高度={info['height']:.3f} pitch={info['pitch']:+.3f} "
                f"fallen={info['fallen']} gait={info['gait']} "
                f"tail_q={np.round(info['tail_q'], 3).tolist()}"
            )
    return True


def check_tail_mobility():
    """尾巴是否真的能动：给尾关节一个恒定目标角，测量实际行程。

    这一步不涉及学习，只验证执行器权限与被动垂坠：
      active  —— 执行器驱动，应能明显改变尾形
      passive —— 力矩为 0，只靠重力/惯性，应有可测摆动
      fixed   —— 刚性锁死，行程应≈0
    """
    hr("4. 尾巴活动权限检查（恒定尾部动作指令）")
    results = {}
    for mode in MODES:
        env = TrexEnv(task="stand", mode=mode, seed=3)
        env.reset(seed=3)
        # 只给尾关节一个固定的非零指令（pitch 交替方向），腿保持 home
        action = np.zeros(env.action_space.shape, np.float32)
        for i, name in enumerate(env.names):
            if name.startswith("tail_"):
                action[i] = 1.0 if name.endswith("_pitch") else 0.0
        tail_q = []
        for _ in range(int(env.duration / env.dt)):
            _, _, term, trunc, _ = env.step(action)
            tail_q.append(env.data.qpos[env.qadr[env.tail_idx]].copy())
            if term or trunc:
                break
        tail_q = np.array(tail_q)
        rom = np.degrees(np.ptp(tail_q, axis=0))
        results[mode] = rom
        print(f"  {mode:7s}: 尾关节行程(deg)={np.round(rom, 2).tolist()}  末端角={np.round(np.degrees(tail_q[-1]), 1).tolist()}")
    print("\n  判读：active 应明显大于 fixed；passive 应有非零行程（重力+惯性），")
    print("        因为尾关节没有预置回中弹簧、也没有做重力补偿。")
    return results


def main():
    (ROOT / "models").mkdir(exist_ok=True)
    from trex import export_models

    export_models()
    check_models()
    ok = check_runs()
    check_zero_action()
    check_tail_mobility()
    hr("自检结论")
    print("模型可运行:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
