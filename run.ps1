# 机器霸王龙项目便捷入口
#
#   .\run.ps1 smoke      物理自检（模型可运行 + 尾巴活动权限）
#   .\run.ps1 bench      吞吐基准
#   .\run.ps1 train      四阶段顺序训练 + 自动评估 + 自动对照（约 2 小时）
#   .\run.ps1 ui         启动交互式查看器（http://127.0.0.1:8770）
#   .\run.ps1 watch      训练进度速览
#   .\run.ps1 report     生成 REPORT_generated.md
#
param([Parameter(Position=0)][string]$cmd = "smoke",
      [int]$Port = 8770,
      [double]$Scale = 1.0)

$ErrorActionPreference = "Stop"
$root = $PSScriptRoot
$py = Join-Path $root ".venv\Scripts\python.exe"
if (-not (Test-Path $py)) { throw "找不到虚拟环境解释器: $py" }

# 沙箱/受限环境下把临时目录与缓存放到工作区
$env:TEMP = Join-Path $root "..\.dsh-tmp"
$env:TMP = $env:TEMP
$env:PYTHONPATH = $root
$env:PYTHONIOENCODING = "utf-8"
New-Item -ItemType Directory -Force -Path $env:TEMP | Out-Null

Set-Location $root

switch ($cmd) {
  "smoke"  { & $py "scripts\smoke.py" }
  "bench"  { & $py "scripts\bench.py" }
  "train"  { & $py "scripts\run_all.py" --scale $Scale --envs 8 --episodes 30 }
  "ui"     { & $py "app.py" --port $Port }
  "watch"  { & $py "scripts\watch.py" }
  "report" { & $py "scripts\report.py" --out REPORT_generated.md }
  default  { Write-Host "用法: .\run.ps1 [smoke|bench|train|ui|watch|report]" }
}
