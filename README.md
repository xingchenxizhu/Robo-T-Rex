# 机器霸王龙 · 尾巴-身体协同 MuJoCo 强化学习实验

用 MuJoCo 搭建一台**可运行的双足机器霸王龙**（躯干、头颈、可开合下颌、双腿脚、**可活动尾巴**），
用 PPO 分四个阶段训练它完成站立平衡、前探咬合、奔跑停止、转向追逐，
并通过**尾巴对照实验**回答一个具体问题：

> 尾巴到底为全身动作贡献了什么？

核心立场（与任务要求一致）：

- **不预设**"身体往左、尾巴一定往右"这类相位关系；
- **不人为添加**固定延迟、随机抖动或过冲来"显得自然"；
- **不奖励**"尾巴必须动起来"；
- 尾巴的"活性"来自**物理**（重力垂坠 + 惯性跟随 + 与躯干的反作用），以及**策略主动使用它**，
  这两者在代码里都是可测量、可对照的，而不是脚本化的动画。

---

## 1. 结论速览（详见 `REPORT.md`）

| 项目 | 状态 |
|---|---|
| 代码实现 | ✅ 模型 / 四阶段环境 / 训练入口 / 评估 / 对照 / 诊断 / 查看器 |
| 仿真已运行 | ✅ 全部任务×三种尾巴模式均已跑通（`scripts/smoke.py`） |
| 尾巴物理权限已验证 | ✅ active 28–34°、passive 46–56°、fixed≈0°（无需训练即可复现） |
| 纯物理对照已验证 | ✅ 零动作 + 扰动下被动尾巴比 PD 夹住的尾巴稳定得多 |
| 阶段一 站立 + 扰动 | ✅ 已收敛（独立种子复测：成功率 1.00、跌倒率 0.00、晃动 0.087 m、恢复 0.46 s） |
| 阶段二 前探咬合收回 | ✅ 已收敛（成功率 1.00、reached/bitten 1.00、回位二次晃动 0.006 rad） |
| 阶段三 奔跑减速停止 | ✅ **已收敛**（成功率 1.00；位移 4.95 m；**11 次腾空**；占空比 0.63<1） |
| 阶段四 转向追逐 | ✅ **已收敛**（成功率 0.83；位移 4.15 m；追上并咬合；腾空占比 0.23） |
| 尾巴对照（同策略消融） | ✅ 已生成三组对照表与图（`evaluations/*/compare.md`） |
| 纯物理对照（无策略） | ✅ 被动尾巴 0% 跌倒、PD 拉住的主动尾巴 33% 跌倒 |

> **达标线**：成功率 ≥ 0.80（写在报告里以便复现判读）。
> 四个阶段都做了**选择/复测种子分离**：先用种子 12345 在所有检查点里选，
> 再用从未参与选择的种子 999 复测并报告
> （`scripts/select_ckpt.py`、`evaluations/ckpt_selection_*.json`）——
> 因为训练期成功率被探索噪声压到 0，`best_model` 选不出好检查点
> （阶段三的 `final_model` 会摔倒，而 900k 检查点成功率 1.00）。
> 两个**人为先验**必须声明：阶段三/四的观测含一个步态时钟（1.1 Hz 相位），
> 见 `REPORT.md` 7.3。
> 调试过程中踩到的十几个"判据/奖励在物理上不可达"的问题全部写进
> `REPORT.md` 第 5 节——它们直接决定了实验能不能得到有意义的结论。

---

## 2. 运行环境（本机实测）

| 项目 | 实测值 |
|---|---|
| 操作系统 | Windows 10 (10.0.19045) |
| 解释器 | `E:\ai_daima\DeepSeek_Harness\机器恐龙\.venv` — **Python 3.11.15** |
| 关键依赖 | mujoco 3.13.0 / gymnasium 1.3.0 / stable-baselines3 2.9.0 / torch 2.14.0+cpu / numpy 2.4.6 |
| GPU | NVIDIA GeForce GTX 1650, 4 GB, 驱动 566.03（CUDA 可用） |
| 训练后端 | **CPU**（见下方说明） |

**为什么用 CPU 而不是 GPU：** 本任务的瓶颈是 MuJoCo 的 CPU 步进（26 自由度、500 Hz、含接触），
PPO 的策略网络只有 128×128 MLP。实测单进程稳定在 **~850 控制步/秒**（3D 任务）。
RTX 1650 只有 4 GB 显存，用 GPU 跑小批量 MLP 反而要付出显存与主机-设备拷贝开销。
若以后要扩到大规模并行环境（MJX），再切 GPU 才划算。

**关于多进程：** 本机沙箱禁止子进程间命名管道，`SubprocVecEnv` 直接抛 `PermissionError`。
因此训练使用 `DummyVecEnv`（单进程串行多环境）。训练入口保留了 `--envs` 参数，
换到不受限的机器上可以无缝切到 `SubprocVecEnv`。

### 安装

```powershell
# 已创建好的虚拟环境
$py = "E:\ai_daima\DeepSeek_Harness\机器恐龙\.venv\Scripts\python.exe"

# 依赖（uv 或 pip 均可；本机沙箱外目录不可写，故缓存放进工作区）
$env:UV_CACHE_DIR = "E:\ai_daima\DeepSeek_Harness\.uv-cache"
uv pip install --python $py mujoco gymnasium stable-baselines3 torch numpy `
    matplotlib flask imageio imageio-ffmpeg tensorboard pandas pillow
```

---

## 3. 模型与物理假设

> 完整列表在 `config.json` 的 `assumptions` 字段里。以下是必须知道的部分。

**这不是真实硬件辨识结果，也不是霸王龙解剖复原**，而是一个**说明性小机器人**：

| 部件 | 质量 | 说明 |
|---|---|---|
| 躯干 | 8.0 kg | 椭球 0.27×0.14×0.16 m，站立髋高 0.785 m |
| 颈 / 头 / 下颌 | 0.7 / 1.3 / 0.25 kg | 下颌铰链 0–35° |
| 大腿 / 小腿 / 脚 | 1.8 / 1.0 / 0.6 kg（每条腿） | 脚为 0.32×0.15×0.10 m 盒体 |
| 尾巴 | 2.4 kg（4 段，总长 1.15 m） | **可配置**，见下节 |
| 整机 | **19.46 kg** | |

其他约定：

1. 惯量由 MuJoCo 按几何体 + 质量自动计算（未用实测惯量张量）。
2. 平面阶段（阶段一/二）的根关节 = 2 个滑移 + 1 个俯仰铰链，**这是声明的实验限制**，不是物理约束。
3. 所有动作只能通过**受限关节执行器 + 物理接触**产生；从不改写 `qpos/qvel`；
   唯一的外力是扰动实验里**显式声明**的 `xfrc_applied`。
4. 地面为刚性平面，`condim=3`，滑动摩擦系数 1.0（脚底 1.2）。

### 3.1 重力补偿：本研究的关键设计决定

- **腿部关节**（髋/膝/踝）施加 `qfrc_bias` 补偿 —— 假设腿部控制器具备基于模型的支撑力矩，这是足式机器人的常规做法。
- **尾巴 / 颈 / 颌不做任何补偿**（`tail.gravity_comp = false`）。

被取消掉的是 `qfrc_bias = C(q,q̇) + g(q)`，即**重力 + 科氏/离心项**。
如果对尾巴也补偿，尾巴会变成"失重"的刚性杆：既不垂坠，也不会因为躯干加速而被甩动，
看起来就是机械的。不做补偿后，尾巴保留：

- **静态垂坠**（零动作下尾巴自然下垂到约 −0.30 rad，见 `scripts/smoke.py` 第 3 项）；
- **惯性跟随**（躯干加速 → 尾部因惯性滞后 → 产生鞭梢效应和反作用力矩）。

这是本研究的**自变量**，三种对照（active/passive/fixed）共享同一套参数，不是给某一组开后门。

---

## 4. 尾巴：可配置，且"活"

尾巴完全由 `config.json` 的 `tail` 段定义，改配置就能换尾巴：

```jsonc
"tail": {
  "segments": 4,               // 关节段数
  "length": 1.15,              // 总长 (m)
  "mass": 2.4,                 // 总质量 (kg)
  "segment_mass_ratio": 0.62,  // 逐段质量衰减比（几何级数）
  "base_radius": 0.075,        // 根部半径，向尾尖 taper 到 30%
  "range_deg": 45,             // 俯仰关节范围 ±45°
  "yaw_range_deg": 35,         // 3D 时的偏航范围
  "torque_proximal": 32.0,     // 根部力矩上限 (N·m)，向尾尖降到 14
  "speed_proximal": 6.0,       // 根部速度上限 (rad/s)，向尾尖放宽到 8
  "stiffness": 0.0,            // 被动回中刚度 = 0（不预置弹簧！）
  "damping": 0.25,             // 被动阻尼（组织/摩擦）
  "gravity_comp": false        // 不做重力/科氏补偿
}
```

### 4.1 尾巴"活性"的三条物理来源（都不是脚本）

1. **没有回中弹簧**：`stiffness = 0`，尾巴不会自动弹回中位。
2. **不做重力补偿**：重力真正作用在尾巴上，它会垂、会摆、会把力矩传回躯干。
3. **策略可以主动驱动它**，但环境**从不**直接命令尾巴目标角——尾巴的每一个动作都来自策略。

### 4.2 尾巴到底有多"活"？`scripts/smoke.py` 的实测（无需训练）

给尾关节一个恒定的非零指令，看实际行程（同一套形态、同一套被动参数）：

| 尾巴模式 | 尾关节行程 (deg) | 说明 |
|---|---|---|
| active（执行器驱动） | 28.5 / 34.2 / 34.1 / 31.6 | 执行器权限充足 |
| passive（力矩恒 0） | 46.4 / 56.0 / 38.2 / 46.0 | **纯重力+惯性就能大幅摆动** |
| fixed（刚性锁死） | ≈0.01 | 对照组，尾巴不动 |

passive 组行程比 active 还大，正是因为尾巴没有被弹簧夹住、也没有被补偿成"失重"。
这与旧版实现（`stiffness=60`、`damping=3`、全关节 `qfrc_bias` 补偿、尾关节 kp 仅 35）
形成对照——那套设置会把尾巴"夹住"，使策略几乎无法使用它。

---

## 5. 三个对照条件（同一形态，只改驱动方式）

| 条件 | 含义 | 实现方式 |
|---|---|---|
| `active` | 尾关节由策略执行器驱动 | 训练时使用的模式 |
| `passive` | 尾关节执行器力矩恒为 0，自由被动摆动 | 每步把尾部力矩清零 |
| `fixed` | 尾关节用 equality 约束刚性锁死 | `data.eq_active` 开关（XML 完全相同） |

- 三者的 **XML 逐字相同**，观测空间与动作空间**完全一致**（12 / 20 维动作），
  `models/*_inertia.json` 里记录了质量、惯量一致性校验。
- `passive` / `fixed` **不单独训练**（按要求只训练阶段一到四）。
  对照是**对同一策略做尾巴驱动方式消融**：固定策略，只改尾巴的执行器权限，看指标怎么变。
  报告里会明确标注这一点及其局限。

---

## 6. 四阶段任务

| 阶段 | task | 维度 | 时长 | 内容 |
|---|---|---|---|---|
| 一 | `stand` | planar | 6.0 s | 站立 → 随机小幅外力扰动 → 观察恢复 |
| 二 | `reach` | planar | 6.5 s | 站稳 → 前探 → 闭颌咬合 → 收回 → **再次站稳** |
| 三 | `loco` | 3D | 11.5 s | 站立 → 行走 → 加速 → 奔跑 → 减速 → 停止 |
| 四 | `chase` | 3D | 15.5 s | 速度/转向指令跟踪 → 追逐移动猎物 → 接近减速 → 咬合 |

**阶段三不是"跑得快就算会跑"**：步态判定基于接触状态而不是速度——

- `walk`：单脚支撑不断切换（`support_switches ≥ 4`）且**没有腾空相**（`flight_steps = 0`）；
- `run`：出现 ≥2 次腾空事件且最长腾空 > 0.05 s；
- 跌倒期间的腾空会被剔除（`airborne_ok` 判据），避免把摔倒误判成奔跑。

**阶段二专门检查"尾巴补偿后的回位是否引起二次晃动"**：`rest` 段的俯仰峰峰值
（`return_sway_rad`）、回稳时间（`return_settle_s`）、以及"尾巴是否回到低能量状态"
（`tail_rest_ok`：尾关节速度范数 < 1.2 rad/s）。

### 6.1 两个必须说明的判据口径

**距离判据用"到目标表面"，不是"到目标中心"。** 猎物是半径 0.075 m 的实体胶囊，
嘴部最多只能贴到它的表面；若用中心距定阈值（例如 <0.07 m），
阈值会小于物理极限，判据永远不可达。因此
`mouth_to_surface = max(0, |嘴 − 目标中心| − 猎物半径)`，
`reached` 阈值 0.06 m、`bitten` 阈值 0.09 m。
`scripts/probe_reach.py` 可以独立验证可达性。

**咬合是"简化咬合"（任务允许的初期简化）。** 头骨几何体的前端面比下颌前缘更靠前
约 1 cm，因此任何目标都会先撞到头骨，要求 `jaw_geom` 接触在几何上不可能发生。
判定改为：嘴部到表面距离 < 0.09 m **且** 存在嘴部接触 **且**
下颌被主动闭合到 < 0.12 rad **且** 接触力在 0.5–60 N。

**"额外迈步"只测量、不否决。** 受扰恢复本来就允许迈步，
因此它作为三组对照的比较指标报告，不作为成功门槛。

**"恢复稳定"是个人为定义**：`|pitch| < 0.15 rad`、`|roll| < 0.12 rad`、
躯干角速度范数 `< 0.4`，并保持 0.3 s（`TrexEnv._is_stable`）。

---

## 7. 观测 / 动作 / 奖励

**观测**（阶段一 70 维 …… 阶段三 98 维）：
关节位置与速度、躯干姿态（重力方向在体系下）、躯干角速度与线速度、高度、
足部接触状态、目标/猎物相对位置与相对速度（体坐标）、
嘴部期望偏差、速度与转向指令、质心相对躯干位置、当前阶段 one-hot、任务 one-hot、上一步动作。

**动作**：每个执行器一路，`[-1,1]`，解释为"相对站立姿态的目标关节角偏移"，经 PD 变成力矩，
再经**力矩上限**和**速度上限**两道限制。**没有任何一路是脚本化轨迹。**

| 动作缩放 | 值 (rad) |
|---|---|
| 尾 pitch / 尾 yaw | 0.60 / 0.50 |
| 腿 hip/knee/ankle | 0.45 |
| 髋 roll / yaw | 0.30 |
| 颈 / 颌 | 0.45 / 0.30 |

**奖励**（每一项都在 `info["reward_terms"]` 里逐项可查）：

| 项 | 权重 | 作用 |
|---|---|---|
| `alive` | +0.5 | 存活 |
| `upright` | +0.8·exp(−4(pitch²+roll²)) | 姿态端正 |
| `height` | +0.6·exp(−30(h−0.785)²) | 保持站立高度 |
| `task` | +1.0·(各阶段任务项) | 任务完成 |
| `energy` | −2e−5·Σ|τ·q̇| | 能耗（**极小**，只抑制无意义抖动） |
| `action_rate` | −0.02·mean(Δa²) | 动作突变（**很小**，不压制必要动作） |
| `slip` | −0.1·足底滑移速度 | 打滑 |
| `target_force` | −0.003·max(0, F−40 N) | 目标接触力过大 |
| `jaw_open` | +0.15·exp(−20(q_jaw−0.25)²) | 非咬合阶段保持张嘴，防止"一直闭嘴"投机 |
| `fall` | −10 | 跌倒终止 |

**没有**任何"尾巴必须动"的奖励项，也**没有**惩罚尾巴速度/行程。
能耗与动作变化惩罚都被压到很小的量级，避免压制必要动作。

**终止条件**：高度 < 0.50 m、|pitch| > 1.0 rad、|roll| > 0.7 rad、
状态非有限、或躯干/头/颌/小腿触地。

---

## 8. 目录结构

```
机器恐龙/
├── config.json               # 唯一的主配置（物理/机器人/尾巴/控制/奖励/PPO/假设）
├── trex/
│   ├── model.py              # MuJoCo XML 构建（可配置尾巴）、模型导出与惯量快照
│   ├── env.py                # 四阶段 Gymnasium 环境 + 三种对照 + 步态/事件判定
│   ├── vec.py                # 环境工厂（DummyVecEnv 用）
│   ├── callbacks.py          # 课程调度 + 训练指标/最优模型
│   ├── rollout.py            # 策略滚动评估
│   └── analysis.py           # 尾巴-身体协同分析（互相关滞后、频谱、丰富度）
├── scripts/
│   ├── smoke.py              # 自检：模型可运行 + 尾巴活动权限 + 形态一致性
│   ├── bench.py              # 吞吐基准
│   ├── train.py              # 训练入口（分阶段课程）
│   ├── evaluate.py           # 单条件评估
│   ├── compare.py            # 尾巴对照实验（表格 + 图）
│   └── preview.py            # 导出画面拼图 / MP4
├── app.py                    # Flask 交互式查看器
├── web/index.html            # 查看器前端（无外部依赖）
├── models/                   # 导出的 6 个 XML + 惯量快照
├── runs/                     # 四个正式实验（TB 事件、检查点、metrics.json）
│   └── sN_*/selected_model.zip  # select_ckpt.py 按确定性评估选出的检查点
├── runs_archive/             # 迭代过程中的历史实验（阶段三 4 轮、阶段四 5 轮）
├── evaluations/              # 评估与对照结果 + ckpt_selection_*.json
├── preview/                  # 画面导出（拼图 PNG + MP4）
└── .gitignore                # 忽略清单（见 8.1）
```

### 8.1 git 忽略规则

`机器恐龙/.gitignore` 的原则是：**源码 / 配置 / 模型 / 报告 / 界面 /
"实际生成的检查点"要上传；虚拟环境、缓存、中间检查点、本地迭代历史不上传。**

实测（`scripts/check_gitignore.py`）：整棵树只会上传 **140 个文件 / 18.99 MB**。

| 分类 | 内容 | 上传 |
|---|---|---|
| 虚拟环境 | `.venv/`（850 MB / 2.6 万文件） | ❌ |
| 缓存 | `__pycache__/`、`.mplcache/`、`.uv-cache/`、`.dsh-tmp/`、`*.log` | ❌ |
| 中间检查点 | `runs/*/checkpoints/`（约 28 MB，每 8–12 万步一个） | ❌ 可由训练复现 |
| 迭代历史 | `runs_archive/**/*.zip`、`runs_archive/*/tb/`（约 73 MB） | ❌ 只留 `train_meta.json` / `metrics.json` |
| 系统垃圾 | `Thumbs.db`、`.DS_Store`、`Desktop.ini`、编辑器目录 | ❌ |
| 源码与文档 | `trex/` `scripts/` `web/` `app.py` `run.ps1` `*.md` `config.json` `requirements.txt` | ✅ |
| 模型 | `models/*.xml`、`models/*_inertia.json` | ✅ |
| **交付检查点** | `runs/*/{selected,best,final}_model.zip` | ✅ |
| 训练记录 | `runs/*/train_meta.json`、`metrics.json`、`tb/`（界面要读） | ✅ |
| 评估与对照 | `evaluations/**` | ✅ |
| 画面导出 | `preview/**`（PNG + MP4） | ✅ |

改完 `.gitignore` 一定要用验证脚本过一遍：

```powershell
& $py scripts\check_gitignore.py            # 假树断言 + 真实项目"会上传什么"报告
& $py scripts\check_gitignore.py --no-real  # 只跑假树断言
```

它在临时目录里搭一棵同名假树建 git 仓库做断言（该挡的挡住、该传的没被误挡），
然后对真实项目做一次 `git add -A -n` 干跑并打印体积分布，
**不会在你的项目里留下 `.git`**。
> 这个脚本第一次跑就抓到了一个真 bug：`.gitignore` 里**行尾**写的 `#` 注释
> 不是注释而是模式的一部分，导致 `.mplcache/`、`.dsh-tmp/` 两条规则完全失效。

> 另外工作区根目录 `E:\ai_daima\DeepSeek_Harness\.gitignore` 只做一件事：
> 挡住根目录的 `.uv-cache/`（实测 **812 MB**）与 `.dsh-tmp/`。
> 如果你不从根目录建仓库，删掉它没有任何影响。**注意根目录下还有十几个别的项目**，
> 所以 git 仓库建议建在 `机器恐龙/` 里面，而不是工作区根。

---

## 9. 命令速查

```powershell
$py = "E:\ai_daima\DeepSeek_Harness\机器恐龙\.venv\Scripts\python.exe"
cd "E:\ai_daima\DeepSeek_Harness\机器恐龙"
$env:PYTHONPATH = "E:\ai_daima\DeepSeek_Harness\机器恐龙"
```

**自检（先跑这个，确认模型可运行 + 尾巴能动）**

```powershell
& $py scripts\smoke.py
```

**分阶段训练**（只训练 active 尾巴；后一阶段尽量热启动上一阶段的 `final_model`）

```powershell
# 一键：四阶段顺序训练 → 自动评估 → 自动尾巴对照（约 2 小时）
& $py scripts\run_all.py --scale 1.0 --envs 8 --episodes 30

# 或逐个跑
& $py scripts\train.py --task stand --steps 800000 --envs 8 --out runs/s1_stand
& $py scripts\train.py --task reach --steps 800000 --envs 8 --out runs/s2_reach `
      --init-from runs/s1_stand/final_model
& $py scripts\train.py --task loco  --steps 1200000 --envs 8 --out runs/s3_loco
& $py scripts\train.py --task chase --steps 900000  --envs 8 --out runs/s4_chase `
      --init-from runs/s3_loco/final_model
```

`run_all.py` 支持 `--from <阶段>` 从中间续跑、`--only <阶段>` 单跑，
以及 `--skip-train` / `--skip-eval`。
缺少的可选依赖或观测维度不匹配时，训练入口会**显式打印原因并回退到从头训练**，
不会静默跳过一个阶段。

**评估单个条件**

```powershell
& $py scripts\evaluate.py --model runs/s1_stand/best_model --task stand `
    --episodes 40 --perturb --trace --out evaluations/s1_stand_active.json
```

**尾巴对照实验**（同一策略 × 三种尾巴驱动方式）

```powershell
& $py scripts\compare.py --model runs/s1_stand/final_model --task stand `
    --episodes 30 --perturb --trace --outdir evaluations/s1_stand
# 不加载策略，只看纯物理形态差异
& $py scripts\compare.py --zero-policy --task stand --episodes 15 --perturb `
    --outdir evaluations/s1_stand_physics
```

**一键重跑全部评估 + 自动生成报告数据**

```powershell
& $py scripts\eval_all.py --ckpt final_model --episodes 30   # 四阶段评估 + 对照
& $py scripts\report.py --out REPORT_generated.md            # 汇总表 + 训练曲线
```

**诊断脚本**（用来判断某个判据/子技能到底能不能做到）

```powershell
& $py scripts\probe_reach.py      # 前探/咬合判据的物理可达性
& $py scripts\probe_bite.py --model runs/s2_reach/selected_model   # 咬合阶段逐项分解
& $py scripts\probe_jaw.py        # 下颌执行器权限与单步响应
& $py scripts\probe_gait.py       # 开环步态探测：物理上能否产生净前进
& $py scripts\probe_loco.py       # 对最新检查点做确定性评估（训练期指标不可信）
& $py scripts\check_env.py        # 四任务环境回归 + 观测维度对齐检查
& $py scripts\watch.py            # 训练进度速览（-v 详细）
& $py scripts\summary_all.py      # 打印四阶段最终结果表
& $py scripts\show_selected.py    # 打印每阶段选中的检查点与选择集排名
& $py scripts\check_gitignore.py  # 验证 .gitignore：假树断言 + 真实项目上传清单
& $py scripts\wait_for.py runs/s3_loco/final_model.zip   # 阻塞等待产物出现
```

**可视化界面的三项自检**

```powershell
& $py app.py --port 8770                       # 启动（浏览器开 http://127.0.0.1:8770）
& $py scripts\verify_ui.py --port 8770         # 各任务 × 各尾巴模式全链路
& $py scripts\stress_sessions.py --port 8770   # 多会话渲染不出全黑帧
& $py scripts\capture_frame.py --port 8770 --task loco `
      --checkpoint runs/s3_loco/selected_model.zip --out preview/ui_capture.jpg
```

> 界面有两个坑已修好、并写进 `app.py` 顶部的文档串：
> **①** Flask 必须 `threaded=False` —— MuJoCo 的 OpenGL 上下文是线程绑定的，
> 多线程会让渲染器失效并返回全黑帧（日志里能看到
> `GLFWError: WGL: Failed to make context current`）；
> **②** 会话必须支持多开 —— 早期版本只保留一个会话，
> 于是"另一个页面或脚本一建会话，当前页面的 `/api/frame` 就 404、
> 视口停在空 `<img>` 上显示全黑"。前端现在遇到会话失效会自动重建。

**导出画面**

```powershell
& $py scripts\preview.py --model runs/s1_stand/best_model --task stand --mode active `
    --perturb --out preview/s1_stand_active --frames 6 --video
```

**交互式查看器**

```powershell
& $py app.py --port 8770      # 然后浏览器打开 http://127.0.0.1:8770
```

查看器支持：加载检查点、播放/暂停/单步/重置、0.05x–2x 慢放、
切换任务与尾巴对照、叠加扰动、实时显示姿态/足部接触/接触点/目标位置/尾关节角度条，
以及从 TensorBoard 事件读取的训练曲线（奖励、成功率、质心晃动、尾巴行程、角动量占比、跌倒率）。

---

## 10. 参考

- MuJoCo 3.13 文档：<https://mujoco.readthedocs.io/>
- Stable-Baselines3 PPO：<https://stable-baselines3.readthedocs.io/>
