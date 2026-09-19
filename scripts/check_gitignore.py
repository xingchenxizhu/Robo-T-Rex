"""验证 .gitignore 是否按预期工作。

做法：在临时目录里搭一棵**与项目同名同结构**的假树（文件都是空的），
把真正的 .gitignore 复制进去，`git init` + `git add -A`，
然后检查：
  * 该上传的（源码/配置/报告/模型/交付检查点/评估/预览）确实进了索引
  * 不该上传的（.venv / 缓存 / 系统垃圾）确实被忽略

**中间检查点与 runs_archive 的期望是策略相关的**：脚本会自己读 .gitignore，
看 `runs/*/checkpoints/` 这条规则是"生效"还是"被注释掉"，据此决定
它们是应该被忽略还是应该被上传——两种策略都能正确断言，不会误报。

不碰真实项目目录，也不在项目里留下 .git。

    .venv\\Scripts\\python.exe scripts\\check_gitignore.py
    .venv\\Scripts\\python.exe scripts\\check_gitignore.py --no-real
"""
from __future__ import annotations

import shutil
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SCRATCH = ROOT.parent / ".dsh-tmp" / "gitignore-check"

# --- 与策略无关：无论怎么选都必须被忽略 ---------------------------------------
ALWAYS_IGNORE = [
    ".venv/Scripts/python.exe",
    ".venv/Lib/site-packages/torch/__init__.py",
    "__pycache__/x.cpython-311.pyc",
    "trex/__pycache__/env.cpython-311.pyc",
    ".mplcache/fontlist-v3.11.0.json",
    ".dsh-tmp/junk.txt",
    "Thumbs.db",
    ".DS_Store",
    "scripts/_scratch.py",
]

# --- 与策略无关：无论怎么选都必须被上传 ---------------------------------------
ALWAYS_KEEP = [
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
    "scripts/check_gitignore.py",
    "web/index.html",
    "models/planar_active.xml",
    "models/planar_active_inertia.json",
    "runs/s1_stand/selected_model.zip",
    "runs/s1_stand/best_model.zip",
    "runs/s1_stand/final_model.zip",
    "runs/s1_stand/train_meta.json",
    "runs/s1_stand/metrics.json",
    "runs/s1_stand/tb/stand_0/events.out.tfevents.1",
    "evaluations/s1_stand/compare.md",
    "evaluations/ckpt_selection_loco.json",
    "preview/s3_loco_running/rollout.mp4",
]
# 注：runs_archive/*/train_meta.json 是否上传取决于归档策略，见下。

# --- 策略相关的规则组 ---------------------------------------------------------
# 每组的期望由 .gitignore 自动判定：规则行**未被注释** => 探针必须被忽略；
# 规则行被注释掉（行首 #）=> 探针必须被上传。这样改策略后脚本不会误报。
POLICY_GROUPS = [
    {
        "label": "中间检查点 runs/*/checkpoints/",
        "rule": "runs/*/checkpoints/",
        "probes": [
            "runs/s1_stand/checkpoints/ckpt_80000_steps.zip",
            "runs/s4_chase/checkpoints/ckpt_560000_steps.zip",
        ],
    },
    {
        "label": "迭代历史 runs_archive/",
        "rule": "runs_archive/**/*.zip",
        "probes": [
            "runs_archive/_s3_loco_v1/best_model.zip",
            "runs_archive/_s3_loco_v1/final_model.zip",
            "runs_archive/_s4_chase_old5/checkpoints/ckpt_100000_steps.zip",
            "runs_archive/_s3_loco_v1/tb/loco_0/events.out.tfevents.1",
            "runs_archive/_s3_loco_v1/train_meta.json",
        ],
    },
    {
        "label": "备份文件 *.bak",
        "rule": "*.bak",
        "probes": ["notes.bak"],
    },
    {
        "label": "备份文件 *.orig",
        "rule": "*.orig",
        "probes": ["patch.orig"],
    },
]


def active_rules() -> set[str]:
    """返回 .gitignore 里**未被注释**的规则行（只有行首的 # 才是注释）。"""
    text = (ROOT / ".gitignore").read_text(encoding="utf-8")
    return {ln.strip() for ln in text.splitlines() if ln.strip() and not ln.startswith("#")}


def build_tree(paths: list[str]) -> None:
    if SCRATCH.exists():
        shutil.rmtree(SCRATCH, ignore_errors=True)
    for rel in paths:
        p = SCRATCH / rel
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(b"x")
    shutil.copyfile(ROOT / ".gitignore", SCRATCH / ".gitignore")


def git(*args: str) -> str:
    r = subprocess.run(["git", *args], cwd=str(SCRATCH), capture_output=True, text=True)
    return (r.stdout or "") + (r.stderr or "")


def real_tree_report() -> int:
    """对真实项目做一次"会上传什么"的报告。

    临时 `git init` → `git add -A -n`（干跑，不写索引/对象）→ 立刻删掉 .git。
    全程 try/finally，不会在项目里留下仓库。
    """
    gitdir = ROOT / ".git"
    if gitdir.exists():
        r = subprocess.run(["git", "ls-files"], cwd=str(ROOT),
                           capture_output=True, text=True)
        files = [f for f in (r.stdout or "").splitlines() if f]
        if not files:
            print("\n项目里已有 .git 但没有跟踪文件，跳过真实树检查")
            return 0
        print("\n项目里已经有 .git —— 直接用 `git ls-files` 报告已跟踪内容")
    else:
        created = False
        try:
            subprocess.run(["git", "init", "-q"], cwd=str(ROOT), capture_output=True, text=True)
            created = True
            r = subprocess.run(["git", "-c", "core.quotepath=false", "add", "-A", "-n"],
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
        size = p.stat().st_size if p.exists() else 0
        total += size
        top = f.split("/")[0] if "/" in f else "(项目根)"
        b = buckets.setdefault(top, [0, 0])
        b[0] += 1
        b[1] += size

    print(f"\n真实项目：{len(files)} 个文件，合计 {total/1024/1024:.2f} MB")
    print(f"{'目录/前缀':<22}{'文件数':>8}{'大小':>13}")
    for k in sorted(buckets, key=lambda x: -buckets[x][1]):
        n, s = buckets[k]
        print(f"{k:<22}{n:>8}{s/1024/1024:>10.2f} MB")
    return 0


def main():
    gi = ROOT / ".gitignore"
    if not gi.exists():
        print("找不到 .gitignore")
        return 1

    rules = active_rules()
    print("当前策略（自动读自 .gitignore，规则未被注释 = 忽略）：")
    for g in POLICY_GROUPS:
        active = g["rule"] in rules
        print(f"  {g['label']:<38} -> {'忽略（不上传）' if active else '上传（规则被注释）'}")

    must_ignore = list(ALWAYS_IGNORE)
    must_keep = list(ALWAYS_KEEP)
    probes = list(ALWAYS_IGNORE) + list(ALWAYS_KEEP)
    for g in POLICY_GROUPS:
        probes.extend(g["probes"])
        target = must_ignore if g["rule"] in rules else must_keep
        target.extend(g["probes"])

    build_tree(probes)
    git("init", "-q")
    git("add", "-A")
    tracked = set(git("ls-files").split())
    print(f"\n假树索引中的文件数：{len(tracked)}")

    bad_uploaded = [p for p in must_ignore if p in tracked]
    bad_missing = [p for p in must_keep if p not in tracked]

    print(f"\n不该上传却被加进去的（{len(bad_uploaded)}）：")
    for p in bad_uploaded:
        print("  ✗", p)
    print(f"该上传却被忽略掉的（{len(bad_missing)}）：")
    for p in bad_missing:
        print("  ✗", p)

    ok = not bad_uploaded and not bad_missing
    print("\n" + ("PASS：.gitignore 行为与当前策略一致" if ok else "FAIL：规则需要调整"))

    if ok and "--no-real" not in sys.argv:
        real_tree_report()
    return 0 if ok else 2


if __name__ == "__main__":
    sys.exit(main())
