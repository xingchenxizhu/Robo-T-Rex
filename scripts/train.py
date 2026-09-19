"""训练入口。

示例：
    .venv\\Scripts\\python.exe scripts\\train.py --task stand --steps 400000 --out runs/s1_stand
    .venv\\Scripts\\python.exe scripts\\train.py --task reach --steps 1500000 --out runs/s2_reach --init-from runs/s1_stand/best_model
    .venv\\Scripts\\python.exe scripts\\train.py --task loco  --steps 3000000 --out runs/s3_loco  --init-from runs/s2_reach/best_model
    .venv\\Scripts\\python.exe scripts\\train.py --task chase --steps 2500000 --out runs/s4_chase --init-from runs/s3_loco/best_model

说明：
  * 只训练主实验（active 尾巴）。fixed / passive 不单独训练，
    而是在评估阶段对同一个策略做"尾巴驱动方式"的消融（见 scripts/compare.py）。
  * 课程参数由 CurriculumCallback 按训练进度在线调整。
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.vec_env import DummyVecEnv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trex import load_config  # noqa: E402
from trex.callbacks import CurriculumCallback, MetricsCallback  # noqa: E402
from trex.vec import make_env  # noqa: E402

# 每个阶段的课程表：(训练进度比例, 要写入 TrexEnv 的属性)
CURRICULA = {
    "stand": [
        (0.00, {"perturb": False, "push_scale": 0.0}),
        (0.25, {"perturb": True, "push_scale": 0.45}),
        (0.50, {"perturb": True, "push_scale": 0.75}),
        (0.75, {"perturb": True, "push_scale": 1.00}),
    ],
    "reach": [
        (0.00, {"curriculum": 0, "target_randomize": 0.00}),
        (0.25, {"curriculum": 0, "target_randomize": 0.03}),
        (0.55, {"curriculum": 0, "target_randomize": 0.06}),
        (0.80, {"curriculum": 0, "target_randomize": 0.10}),
    ],
    "loco": [
        # 从"极慢速"起步：指令速度小的时候，哪怕只是小幅前倾/蹭步也能拿满前进奖励，
        # 策略才有机会先学会"往前动"这件事，再逐步把速度拉开。
        (0.00, {"speed_scale": 0.20}),
        (0.15, {"speed_scale": 0.30}),
        (0.30, {"speed_scale": 0.45}),
        (0.45, {"speed_scale": 0.60}),
        (0.62, {"speed_scale": 0.78}),
        (0.80, {"speed_scale": 0.90}),
        (0.92, {"speed_scale": 1.00}),
    ],
    "chase": [
        (0.00, {"curriculum": 0}),
        (0.25, {"curriculum": 1}),
        (0.50, {"curriculum": 2}),
        (0.75, {"curriculum": 3}),
    ],
}

BEST_KEY = {
    "stand": "success_rate",
    "reach": "success_rate",
    # 奔跑需要腾空相证据；若尚未学会，用"会移动并停住"作为退而求其次的判据，
    # 否则 best_model 会退化成窗口刚满时的那个早期检查点。
    "loco": [("success_locomotion", 1.0), ("success_run", 0.6), ("fall_rate", -0.5)],
    "chase": [("success_rate", 1.0), ("success_approach", 0.5),
              ("bitten", 0.5), ("fallen", -0.5)],
}

# 折扣因子：站立只需短时视界；前探/奔跑/追逐需要更长的信用分配
GAMMA = {"stand": 0.99, "reach": 0.995, "loco": 0.995, "chase": 0.995}

# 初始探索强度（log 标准差）。实测：std>0.16 时站立会被纯噪声掀翻，
# 训练指标会失真；因此站立阶段必须用较小探索，运动阶段逐步放宽。
LOG_STD_INIT = {"stand": -2.3, "reach": -2.0, "loco": -0.9, "chase": -0.9}


def parse_args():
    p = argparse.ArgumentParser(description="机器霸王龙 PPO 训练")
    p.add_argument("--task", required=True, choices=["stand", "reach", "loco", "chase"])
    p.add_argument("--mode", default="active", choices=["active", "passive", "fixed"])
    p.add_argument("--steps", type=float, default=5e5)
    p.add_argument("--envs", type=int, default=8)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    p.add_argument("--init-from", default=None, help="从已有检查点热启动")
    p.add_argument("--no-curriculum", action="store_true")
    p.add_argument("--n-steps", type=int, default=1024)
    p.add_argument("--batch-size", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--ent-coef", type=float, default=None)
    p.add_argument("--net", type=int, nargs=2, default=None)
    p.add_argument("--gamma", type=float, default=None)
    p.add_argument("--curriculum-offset", type=int, default=0,
                   help="课程起点（例如已训练过一半则从中间开始）")
    return p.parse_args()


def main():
    args = parse_args()
    cfg = load_config()
    total = int(args.steps)
    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.mkdir(parents=True, exist_ok=True)

    ppo_cfg = cfg["ppo"]
    net = tuple(args.net) if args.net else tuple(ppo_cfg["net_arch"])
    ent = args.ent_coef if args.ent_coef is not None else ppo_cfg["ent_coef"]
    gamma = args.gamma if args.gamma is not None else GAMMA[args.task]
    log_std_init = float(LOG_STD_INIT[args.task])

    venv = DummyVecEnv([
        make_env(args.task, args.mode, args.seed, i) for i in range(args.envs)
    ])

    callbacks = []
    metrics = MetricsCallback(
        log_dir=out, window=60, save_best=True,
        best_key=BEST_KEY[args.task], print_every=25, verbose=1,
    )
    callbacks.append(metrics)
    callbacks.append(CheckpointCallback(
        save_freq=max(1, total // 10 // args.envs), save_path=str(out / "checkpoints"),
        name_prefix="ckpt",
    ))
    if not args.no_curriculum:
        callbacks.append(CurriculumCallback(CURRICULA[args.task], total, verbose=1))

    model = None
    if args.init_from:
        init_path = Path(args.init_from)
        if not init_path.is_absolute():
            init_path = ROOT / init_path
        if init_path.suffix != ".zip" and (init_path.with_suffix(".zip")).exists():
            init_path = init_path.with_suffix(".zip")
        print(f"[init] 尝试从 {init_path} 热启动")
        try:
            model = PPO.load(str(init_path), env=venv, learning_rate=args.lr,
                             n_steps=args.n_steps, batch_size=args.batch_size,
                             ent_coef=ent, gamma=gamma,
                             tensorboard_log=str(out / "tb"), seed=args.seed)
            import torch

            with torch.no_grad():
                model.policy.log_std.data.fill_(log_std_init)
            model.verbose = 1
            print("[init] 热启动成功（已重置探索强度）")
        except ValueError as exc:
            # 观测/动作空间不匹配（例如平面阶段 -> 三维阶段，自由度数不同）
            print(f"[init] 空间不匹配，改为从头训练：{exc}")
            model = None

    if model is None:
        model = PPO(
            "MlpPolicy", venv,
            n_steps=args.n_steps,
            batch_size=args.batch_size,
            n_epochs=ppo_cfg["n_epochs"],
            learning_rate=args.lr,
            gamma=gamma,
            gae_lambda=ppo_cfg["gae_lambda"],
            clip_range=ppo_cfg["clip_range"],
            ent_coef=ent,
            max_grad_norm=ppo_cfg["max_grad_norm"],
            policy_kwargs={"net_arch": list(net), "log_std_init": log_std_init},
            tensorboard_log=str(out / "tb"),
            device=ppo_cfg["device"],
            seed=args.seed,
            verbose=1,
        )

    meta = {
        "task": args.task, "mode": args.mode, "steps": total, "envs": args.envs,
        "seed": args.seed, "n_steps": args.n_steps, "batch_size": args.batch_size,
        "lr": args.lr, "ent_coef": ent, "net_arch": list(net), "gamma": gamma,
        "log_std_init": log_std_init,
        "curriculum": CURRICULA[args.task] if not args.no_curriculum else None,
        "init_from": args.init_from,
        "obs_dim": int(venv.observation_space.shape[0]),
        "act_dim": int(venv.action_space.shape[0]),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    t0 = time.time()
    # 每个阶段都是独立实验、独立 TB 目录、独立课程表，因此计数从头开始；
    # 否则热启动时 num_timesteps 会带上上一阶段的步数，本阶段预算被腰斩。
    model.learn(total_timesteps=total, callback=callbacks, tb_log_name=args.task,
                reset_num_timesteps=True, progress_bar=False)
    dt = time.time() - t0

    model.save(str(out / "final_model"))
    metrics.dump(out / "metrics.json")
    meta["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    meta["wall_seconds"] = dt
    meta["steps_per_second"] = total / max(1e-9, dt)
    (out / "train_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n[done] {args.task} {total} 步, 用时 {dt/60:.1f} 分钟, {total/dt:.0f} 步/秒")
    print(f"       最优成功率(窗口) = {metrics.best}")
    print(f"       输出目录 = {out}")
    venv.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
