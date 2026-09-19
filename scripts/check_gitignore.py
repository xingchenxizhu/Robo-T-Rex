"""验证 .gitignore 是否按预期工作。

做法：在临时目录里搭一棵**与项目同名同结构**的假树（文件都是空的），
把真正的 .gitignore 复制进去，`git init` + `git add -A`，
然后检查：
  * 该上传的（源码/配置/报告/模型/选中的检查点/评估/预览）确实进了索引
  * 不该上传的（.venv / 缓存 / 中间检查点 / 归档检查点 / 系统垃圾）确实被忽略

不碰真实项目目录，也不在项目里留下 .git。

    .venv\\Scripts\\python.exe scripts\\check_gitignore.py
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT.parent / ".dsh-tmp" / "gitignore-check"

# 必须被忽略（不上传）
MUST_IGNORE = [
    ".venv/Scripts/python.exe",
    ".venv/Lib/site-packages/torch/__init__.py",
    "__pycache__/x.cpython-311.pyc",
    "trex/__pycache__/env.cpython-311.pyc",
    ".mplcache/fontlist-v3.11.0.json",
    ".dsh-tmp/junk.txt",
    "runs/s1_stand/checkpoints/ckpt_80000_steps.zip",
    "runs/s4_chase/checkpoints/ckpt_560000_steps.zip",
    "runs_archive/_s3_loco_v1/best_model.zip",
    "runs_archive/_s3_loco_v1/final_model.zip",
    "runs_archive/_s4_chase_old5/checkpoints/ckpt_100000_steps.zip",
    "runs_archive/_s3_loco_v1/tb/loco_0/events.out.tfevents.1",
    "Thumbs.db",
    ".DS_Store",
    "scripts/_scratch.py",
    "notes.bak",
]

# 必须被上传（进了索引）
MUST_KEEP = [
    "app.py",
    "run.ps1",
    "config.json",
    "requirements.txt",
    "README.md",
    "REPORT.md",
    "REPORT_generated.md",
    ".gitignore",
    "trex/env.py",
    "trex/model.py",
    "scripts/train.py",
    "scripts/run_all.py",
    "scripts/select_ckpt.py",
    "scripts/compare.py",
    "web/index.html",
    "models/planar_active.xml",
    "models/planar_active_inertia.json",
    "runs/s1_stand/selected_model.zip",
    "runs/s1_stand/best_model.zip",
    "runs/s1_stand/final_model.zip",
    "runs/s1_stand/train_meta.json",
    "runs/s1_stand/metrics.json",
    "runs/s1_stand/tb/stand_0/events.out.tfevents.1",
    "runs_archive/_s3_loco_v1/train_meta.json",
    "runs_archive/_s3_loco_v1/metrics.json",
    "evaluations/s1_stand/compare.md",
    "evaluations/ckpt_selection_loco.json",
    "preview/s3_loco_running/rollout.mp4",
    "preview/s3_loco_running/frames.png",
]


def build_tree() -> None:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH, ignore_errors=True)
    for rel in MUST_IGNORE + MUST_KEEP:
        p = SCRATCH / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    shutil.copyfile(ROOT / ".gitignore", SCRATCH / ".gitignore")


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=str(SCRATCH), capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def real_tree_report() -> int:
    """对真实项目做一次"会上传什么"的报告。

    做法：临时 `git init` → `git add -A -n`（干跑，不写索引/对象）→
    立刻删掉刚建的 .git。全程 try/finally，不会在项目里留下仓库。
    """
    gitdir = ROOT / ".git"
    if gitdir.exists():
        print("项目里已经有 .git 了，跳过真实树检查（避免动到你的仓库）")
        return 0
    created = False
    try:
        subprocess.run(["git", "init", "-q"], cwd=str(ROOT), capture_output=True, text=True)
        created = True
        r = subprocess.run(
            ["git", "-c", "core.quotepath=false", "add", "-A", "-n"],
            cwd=str(ROOT), capture_output=True, text=True)
        lines = [ln for ln in (r.stdout or "").splitlines() if ln.startswith("add '")]
        files = [ln[len("add '"):-1] for ln in lines]
    finally:
        if created:
            shutil.rmtree(gitdir, ignore_errors=True)

    total = 0
    buckets: dict[str, list[int]] = {}
    for f in files:
        p = ROOT / f
        try:
            size = p.stat().st_size
        except OSError:
            size = 0
        total += size
        top = f.split("/")[0] if "/" in f else "(项目根)"
        b = buckets.setdefault(top, [0, 0])
        b[0] += 1
        b[1] += size

    print(f"\n真实项目：会上传 {len(files)} 个文件，合计 {total/1024/1024:.2f} MB")
    print(f"{'目录/前缀':<24}{'文件数':>8}{'大小':>12}")
    for k in sorted(buckets, key=lambda x: -buckets[x][1]):
        n, s = buckets[k]
        print(f"{k:<24}{n:>8}{s/1024/1024:>10.2f} MB")
    big = [f for f in files if (ROOT / f).stat().st_size > 3 * 1024 * 1024] \
        if all((ROOT / f).exists() for f in files) else []
    if big:
        print("\n超过 3 MB 的单文件（确认一下是否真要上传）：")
        for f in big:
            print(f"  {f}  {(ROOT / f).stat().st_size/1024/1024:.2f} MB")
    return 0


def main():
    if not (ROOT / ".gitignore").exists():
        print("找不到 .gitignore")
        return 1
    build_tree()
    git("init", "-q")
    git("add", "-A")
    tracked = set(git("ls-files").split())
    print(f"索引中的文件数：{len(tracked)}")

    bad_uploaded = [p for p in MUST_IGNORE if p in tracked]
    bad_missing = [p for p in MUST_KEEP if p not in tracked]

    print(f"\n不该上传却被加进去的（{len(bad_uploaded)}）：")
    for p in bad_uploaded:
        print("  ✗", p)
    print(f"\n该上传却被忽略掉的（{len(bad_missing)}）：")
    for p in bad_missing:
        print("  ✗", p)

    ok = not bad_uploaded and not bad_missing
    print("\n" + ("PASS：.gitignore 行为符合预期" if ok else "FAIL：规则需要调整"))
    print(f"（验证用的假树在 {SCRATCH}，可以随时删掉）")

    if ok and "--no-real" not in sys.argv:
        real_tree_report()
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
