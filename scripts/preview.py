"""把策略跑出来的画面导出成 PNG 拼图与 MP4，用于肉眼检查。

示例：
    .venv\\Scripts\\python.exe scripts\\preview.py --task stand --out preview/s1_stand --mode active
    .venv\\Scripts\\python.exe scripts\\preview.py --model runs/s1_stand/best_model --task stand ^
        --mode passive --out preview/s1_passive --video
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trex import TrexEnv  # noqa: E402


def resolve(p):
    p = Path(p)
    return p if p.is_absolute() else ROOT / p


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None)
    ap.add_argument("--task", required=True, choices=["stand", "reach", "loco", "chase"])
    ap.add_argument("--mode", default="active", choices=["active", "passive", "fixed"])
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--perturb", action="store_true")
    ap.add_argument("--curriculum", type=int, default=0)
    ap.add_argument("--speed-scale", type=float, default=None)
    ap.add_argument("--video", action="store_true")
    ap.add_argument("--seed", type=int, default=3)
    args = ap.parse_args()

    env = TrexEnv(task=args.task, mode=args.mode, seed=args.seed,
                  perturb=args.perturb, curriculum=args.curriculum)
    if args.speed_scale is not None:
        env.speed_scale = args.speed_scale
    model = None
    if args.model:
        from stable_baselines3 import PPO

        p = resolve(args.model)
        if p.suffix != ".zip" and p.with_suffix(".zip").exists():
            p = p.with_suffix(".zip")
        model = PPO.load(str(p), env=env, device="cpu")

    obs, _ = env.reset(seed=args.seed)
    total = int(env.duration / env.dt)
    shots = []
    interval = max(1, total // args.frames)
    video_frames = []
    for i in range(total):
        if model is None:
            action = np.zeros(env.action_space.shape, dtype=np.float32)
        else:
            action, _ = model.predict(obs, deterministic=True)
        obs, _, term, trunc, info = env.step(action)
        if i % interval == 0 and len(shots) < args.frames:
            img = env.render()
            shots.append(img.copy())
            if args.video:
                video_frames.append(img.copy())
        elif args.video:
            video_frames.append(env.render().copy())
        if term or trunc:
            break

    outdir = resolve(args.out)
    outdir.mkdir(parents=True, exist_ok=True)

    h, w, _ = shots[0].shape
    cols = min(4, len(shots))
    rows = int(np.ceil(len(shots) / cols))
    sheet = Image.new("RGB", (cols * w, rows * h), (10, 14, 20))
    for k, img in enumerate(shots):
        sheet.paste(Image.fromarray(img), ((k % cols) * w, (k // cols) * h))
    sheet_path = outdir / "frames.png"
    sheet.save(sheet_path)
    print(f"[written] {sheet_path}  ({cols}x{rows} 拼图, 每 {interval*env.dt:.2f}s 一帧)")

    if args.video and video_frames:
        try:
            import imageio.v2 as iio

            vid = outdir / "rollout.mp4"
            iio.mimsave(str(vid), video_frames, fps=int(round(1 / env.dt / 2)))
            print(f"[written] {vid}  ({len(video_frames)} 帧)")
        except Exception as exc:  # noqa: BLE001
            print(f"[video skipped] {type(exc).__name__}: {exc}")

    print(f"末状态: 步态={info.get('gait')} 时间={info.get('time'):.2f}s "
          f"尾关节={np.round(info.get('tail_q', []), 3).tolist()}")
    env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
