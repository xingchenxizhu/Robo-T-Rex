"""机器霸王龙四阶段 Gymnasium 环境。

任务（task）：
  stand  阶段一 站立与保持平衡（可选外力扰动）
  reach  阶段二 站稳 → 前探 → 咬合并收回 → 再次站稳
  loco   阶段三 站立 → 行走 → 加速 → 奔跑 → 减速停止
  chase  阶段四 跟踪速度/转向指令 → 追逐移动猎物 → 接近减速 → 咬合

对照（mode）：
  active / passive / fixed —— 观测与动作空间三者完全相同，
  唯一差别是尾关节是否受执行器驱动（fixed 通过 equality 约束刚性锁死）。

设计约束（见 config.json 的 assumptions）：
  * 不直接改写 qpos/qvel；一切运动来自受限执行器与接触。
  * 不做尾关节的重力/科氏补偿 → 尾巴保留垂坠与惯性跟随。
  * 没有任何脚本化的摆尾轨迹；环境从不直接命令尾巴目标角。
  * 不奖励"尾巴必须动"。
  * ep_step 内不分配大对象：统计量用滚动累加，角动量用 mj_subtreeVel。
"""

from __future__ import annotations

import copy

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium import spaces

from .model import build_xml, load_config

TASKS = ("stand", "reach", "loco", "chase")
MODES = ("active", "passive", "fixed")
DIMENSION_FOR_TASK = {"stand": "planar", "reach": "planar", "loco": "3d", "chase": "3d"}

PHASES = {
    "stand": [("settle", 1.5), ("hold", 4.5)],
    "reach": [("settle", 1.0), ("probe", 1.6), ("bite", 1.0), ("retract", 1.2), ("rest", 1.7)],
    "loco": [("stand", 1.0), ("walk", 2.0), ("accel", 2.0), ("run", 2.5), ("decel", 2.0), ("stop", 2.0)],
    "chase": [("settle", 0.8), ("track", 1.5), ("chase", 9.0), ("attack", 2.5), ("recover", 1.7)],
}
PHASE_DUR = {task: {n: d for n, d in ph} for task, ph in PHASES.items()}

SPEED_PROFILE = {"stand": 0.0, "walk": 0.45, "accel": None, "run": 1.20, "decel": None, "stop": 0.0}

ACTION_SCALE = {
    "tail_pitch": 0.60, "tail_yaw": 0.50, "leg": 0.45,
    "hip_roll": 0.30, "hip_yaw": 0.30, "neck": 0.45, "jaw": 0.45,
}

# 阶段编码固定占用这么多槽位。这样"平面阶段之间"（stand/reach）与
# "三维阶段之间"（loco/chase）的观测维度一致，可以跨阶段热启动策略；
# 平面↔三维因为自由度不同（12 vs 20 动作）本就必须分别训练。
PHASE_SLOTS = 6


def _smoothstep(a: float) -> float:
    a = min(1.0, max(0.0, a))
    return a * a * (3.0 - 2.0 * a)


class TrexEnv(gym.Env):
    metadata = {"render_modes": ["rgb_array"], "render_fps": 50}

    def __init__(
        self,
        task: str = "stand",
        mode: str = "active",
        config: dict | None = None,
        dimension: str | None = None,
        render_mode: str | None = None,
        perturb: bool = False,
        seed: int | None = None,
        record_trace: bool = False,
        curriculum: int = 0,
    ):
        assert task in TASKS, task
        assert mode in MODES, mode
        self.cfg = copy.deepcopy(config or load_config())
        self.task = task
        self.mode = mode
        self.dimension = dimension or DIMENSION_FOR_TASK[task]
        self.render_mode = render_mode
        self.perturb = bool(perturb)
        self.curriculum = int(curriculum)
        self.record_trace = bool(record_trace)
        # 课程学习期间可在线调整
        self.speed_scale = 1.0
        self.target_randomize = float(self.cfg["target"]["randomize"])
        self.push_scale = 1.0   # 扰动强度缩放（课程从 0.45 逐步放到 1.0）
        self.gait_hz = float(self.cfg.get("gait_clock_hz", 1.1))

        xml, meta = build_xml(self.cfg, self.dimension, self.mode)
        self.xml = xml
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.data = mujoco.MjData(self.model)
        self.meta = meta

        self.names = list(meta["names"])
        self.n = len(self.names)
        self.jids = np.array([self.model.joint(n).id for n in self.names])
        self.qadr = self.model.jnt_qposadr[self.jids]
        self.vadr = self.model.jnt_dofadr[self.jids]
        self.torque_limits = meta["torques"].copy()
        self.speed_limits = meta["speeds"].copy()

        limited = self.model.jnt_limited[self.jids].astype(bool)
        self.q_lo = np.where(limited, self.model.jnt_range[self.jids, 0], -3.0)
        self.q_hi = np.where(limited, self.model.jnt_range[self.jids, 1], 3.0)

        self.is_tail = np.array([n.startswith("tail_") for n in self.names])
        self.is_leg = np.array([n.startswith("left_") or n.startswith("right_") for n in self.names])
        self.is_jaw = np.array([n == "jaw" for n in self.names])
        self.is_neck = np.array([n == "neck" for n in self.names])
        self.tail_idx = np.flatnonzero(self.is_tail)
        self.tail_mask = self.is_tail
        self.n_tail = int(self.is_tail.sum())
        self.jaw_idx = int(np.flatnonzero(self.is_jaw)[0])

        self.home = np.zeros(self.n)
        # 下颌中立位 = 闭合（真实兽脚类的颌本来就闭合，张嘴才是主动行为）。
        # 若把中立位设成半张（0.22 rad），策略必须"连续多步输出大幅闭颌指令"
        # 才能咬合，而逐步独立采样的小方差探索永远发现不了这种持续动作。
        self.home[self.jaw_idx] = 0.0
        scale = np.zeros(self.n)
        for i, name in enumerate(self.names):
            if name.startswith("tail_"):
                scale[i] = ACTION_SCALE["tail_pitch"] if name.endswith("_pitch") else ACTION_SCALE["tail_yaw"]
            elif name.endswith("_hip_roll"):
                scale[i] = ACTION_SCALE["hip_roll"]
            elif name.endswith("_hip_yaw"):
                scale[i] = ACTION_SCALE["hip_yaw"]
            elif name == "neck":
                scale[i] = ACTION_SCALE["neck"]
            elif name == "jaw":
                scale[i] = ACTION_SCALE["jaw"]
            else:
                scale[i] = ACTION_SCALE["leg"]
        self.action_scale = scale

        c = self.cfg["control"]
        self.kp = np.full(self.n, c["kp_leg"])
        self.kd = np.full(self.n, c["kd_leg"])
        self.kp[self.is_tail] = c["kp_tail"]
        self.kd[self.is_tail] = c["kd_tail"]
        self.kp[self.is_neck] = c["kp_neck"]
        self.kd[self.is_neck] = c["kd_neck"]
        self.kp[self.is_jaw] = c["kp_jaw"]
        self.kd[self.is_jaw] = c["kd_jaw"]

        self.gravity_comp = np.zeros(self.n, dtype=bool)
        if c.get("gravity_comp_legs", True):
            self.gravity_comp[self.is_leg] = True
        if bool(self.cfg["tail"].get("gravity_comp", False)):
            self.gravity_comp[self.is_tail] = True
        self.gc_idx = np.flatnonzero(self.gravity_comp)
        self.gc_any = bool(self.gravity_comp.any())

        p = self.cfg["physics"]
        self.dt = float(p["control_dt"])
        self.substeps = int(round(self.dt / self.model.opt.timestep))
        assert abs(self.substeps * self.model.opt.timestep - self.dt) < 1e-9

        # ---- 关键 id ----
        self.torso_id = self.model.body("torso").id
        self.mouth_site = self.model.site("mouth").id
        self.bite_site = self.model.site("bite_point").id
        self.prey_body = self.model.body("prey").id
        self.prey_mocap = int(self.model.body("prey").mocapid[0])
        self.prey_geom = self.model.geom("prey_geom").id
        self.prey_radius = float(self.cfg["target"]["radius"])
        self.floor_geom = self.model.geom("floor").id
        self.jaw_geom = self.model.geom("jaw_geom").id
        self.skull_geom = self.model.geom("skull").id
        self.foot_geoms = [self.model.geom(f"{s}_foot").id for s in ("left", "right")]
        self.foot_bodies = [self.model.body(f"{s}_foot").id for s in ("left", "right")]

        bad = self.cfg["termination"]["bad_ground_bodies"]
        self.bad_ground_geoms = set()
        for b in bad:
            bid = self.model.body(b).id
            for k in range(int(self.model.body_geomnum[bid])):
                self.bad_ground_geoms.add(int(self.model.body_geomadr[bid]) + k)

        # ---- 尾巴身体（角动量统计用，一次算好） ----
        n_seg = int(self.cfg["tail"]["segments"])
        self.tail_body_ids = np.array([self.model.body(f"tail_{i}").id for i in range(n_seg)], dtype=int)
        self.tail_body_mass = self.model.body_mass[self.tail_body_ids].copy()
        self.total_mass = float(self.model.body_mass.sum())

        # ---- 空间 ----
        self.action_space = spaces.Box(-1.0, 1.0, (self.n,), np.float32)
        self.phase_names = [name for name, _ in PHASES[task]]
        self.n_phase = len(self.phase_names)
        self.duration = float(sum(d for _, d in PHASES[task]))
        self.phase_edges = np.cumsum([d for _, d in PHASES[task]])

        # ---- 复用缓冲区 ----
        self._tau = np.zeros(self.n)
        self._qa = np.zeros(self.n)
        self._tmp = np.zeros(self.n)
        self._cf = np.zeros(6)
        self._momentum_buf = np.zeros(3)
        dummy, _ = self.reset(seed=seed)
        self.observation_space = spaces.Box(-np.inf, np.inf, dummy.shape, np.float32)
        self._obs_buf = np.zeros(dummy.shape, np.float32)
        self._obs_buf[:] = dummy
        # reset 之后重新绑定 data 视图（mj_resetData 不换缓冲，但保持一次绑定最稳）
        self._bind_views()

        self.renderer = None
        self.camera = None

    # ------------------------------------------------------------------ #
    def _bind_views(self):
        d = self.data
        self._qpos = d.qpos
        self._qvel = d.qvel
        self._ctrl = d.ctrl
        self._bias = d.qfrc_bias
        self._xfrc = d.xfrc_applied

    # ------------------------------------------------------------------ #
    # 阶段/姿态/速度
    # ------------------------------------------------------------------ #
    def phase_index(self) -> int:
        idx = int(np.searchsorted(self.phase_edges, self.elapsed, side="right"))
        return min(idx, self.n_phase - 1)

    def phase_name(self) -> str:
        return self.phase_names[self.phase_index()]

    def phase_t(self) -> float:
        i = self.phase_index()
        start = 0.0 if i == 0 else float(self.phase_edges[i - 1])
        return self.elapsed - start

    def posture(self):
        R = self.data.xmat[self.torso_id].reshape(3, 3)
        return (
            float(np.arctan2(-R[2, 0], np.hypot(R[0, 0], R[1, 0]))),
            float(np.arctan2(R[2, 1], R[2, 2])),
            float(np.arctan2(R[1, 0], R[0, 0])),
        )

    def torso_velocity(self) -> np.ndarray:
        v = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, self.torso_id, v, 0)
        return v

    def _gravity_body(self) -> np.ndarray:
        R = self.data.xmat[self.torso_id].reshape(3, 3)
        return R.T @ np.array([0.0, 0.0, -1.0])

    # ------------------------------------------------------------------ #
    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        d = self.data
        mujoco.mj_resetData(self.model, d)
        d.qpos[self.qadr] = self.home
        d.qpos[self.qadr[self.is_tail]] = 0.0
        d.eq_active[:] = 1 if self.mode == "fixed" else 0
        mujoco.mj_forward(self.model, d)
        self._bind_views()

        self.initial_mouth = d.site_xpos[self.mouth_site].copy()
        self.initial_x = float(d.xpos[self.torso_id, 0])
        self._setup_prey()

        rng = self.np_random
        self.pushes = []
        if self.perturb:
            pc = self.cfg["perturb"]
            t = float(pc["first_push_time"])
            while t < self.duration - 0.8:
                self.pushes.append(
                    (t, float(rng.uniform(0.6, 1.4)) * float(pc["push_force"])
                     * float(self.push_scale) * float(rng.choice([-1.0, 1.0])))
                )
                t += float(pc["min_gap"]) + float(rng.uniform(0.0, 1.2))
        self.push_end = float(self.cfg["perturb"]["push_duration"])

        self.last_action = np.zeros(self.n)
        self.elapsed = 0.0
        self.steps = 0
        self.energy = 0.0
        self.peak_torque = 0.0
        self.saturated = 0
        self.nctrl = 0
        self.slip_total = 0.0
        self.peak_prey_force = 0.0
        self.fallen = False

        self.reached = False
        self.bitten = False
        self.stood_again = False
        self.flight_events = 0
        self.run_flight_events = 0
        self.flight_steps = 0
        self.run_flight_steps = 0
        self.support_switches = 0
        self.last_single_support = None
        self.extra_steps = 0
        self._foot_air = np.zeros(2)
        self._air_bonus = 0.0
        self.airtime = 0.0
        self.max_airtime = 0.0
        self.duty = np.zeros(2)
        self.recovery_time = None
        self.recovery_hold = 0.0
        self.return_settle = None
        self.attack_time = None
        self.tail_rest_ok = False

        # 滚动统计
        self._com0_xy = d.subtree_com[0][:2].copy()
        self._sway_max = 0.0
        self._pitch_ss = 0.0
        self._roll_ss = 0.0
        self._tail_min = np.full(self.n_tail, np.inf)
        self._tail_max = np.full(self.n_tail, -np.inf)
        self._rest_pitch_min = np.inf
        self._rest_pitch_max = -np.inf
        self._rest_start = None
        self._track_sum = 0.0
        self._track_n = 0
        self._mom_tail_sum = 0.0
        self._mom_total_sum = 0.0
        self._mom_share_n = 0
        self._dist_max = 0.0
        self._speed_max = 0.0
        self._yaw_err_sum = 0.0
        self._yaw_travel = 0.0
        self._prev_yaw = None
        self._last_track_err = 0.0

        self._prey_velocity = np.zeros(3)
        self._prev_prey_pos = self.prey_pos.copy()

        self._contacts = np.zeros(2)
        self._prey_contact = False
        self._jaw_force = 0.0
        self._prey_force = 0.0
        self._bad_ground_now = False
        self._last_vel = None

        self.trace = None
        if self.record_trace:
            keys = ("t", "com_x", "com_y", "pitch", "roll", "yaw", "speed", "cmd_speed",
                    "yaw_rate", "cmd_yaw_rate", "height", "contact_l", "contact_r",
                    "tail_q", "tail_qd", "prey_dist", "mouth_err", "reward")
            self.trace = {k: [] for k in keys}

        self._update_contacts()
        return self._obs(), {}

    def _setup_prey(self):
        t = self.cfg["target"]
        level = min(self.curriculum, 6)
        if self.task == "reach":
            rand = self.target_randomize * (1.0 + level)
            dx = float(t["start_distance"]) + float(self.np_random.uniform(0, rand))
            dz = float(self.np_random.uniform(-rand, rand)) * 0.5
            pos = self.initial_mouth + np.array([dx, 0.0, float(t["reach_height"]) + dz])
        elif self.task == "chase":
            # 从"嘴"起算的初始间距，否则猎物会一开局就进入咬合范围，接近阶段被架空
            d = float(self.cfg["prey"]["start_gap"]) + 0.6 * level
            pos = np.array([self.initial_mouth[0] + d, 0.0, 0.95])
            self.prey_phase = float(self.np_random.uniform(0, 2 * np.pi))
            self.prey_speed = float(self.cfg["prey"]["speed"]) * (1.0 + 0.15 * level)
            self.prey_wander = float(self.cfg["prey"]["turn_rate"]) * (1.0 + 0.5 * level)
        else:
            pos = np.array([0.0, 0.0, -5.0])
        self.prey_pos = np.asarray(pos, dtype=float)
        self.prey_target = self.prey_pos.copy()
        self.data.mocap_pos[self.prey_mocap] = self.prey_pos
        collidable = 1 if self.task in ("reach", "chase") else 0
        self.model.geom_contype[self.prey_geom] = collidable
        self.model.geom_conaffinity[self.prey_geom] = collidable
        # mocap 位置与碰撞开关改完后必须前向一次，否则 reset 返回的观测里目标位姿是旧的
        mujoco.mj_forward(self.model, self.data)

    # ------------------------------------------------------------------ #
    def _move_prey(self):
        if self.task != "chase":
            return
        t = self.elapsed - PHASE_DUR["chase"]["settle"]
        if t < 0:
            return
        level = min(self.curriculum, 6)
        self.prey_pos = np.array([
            self.initial_mouth[0] + float(self.cfg["prey"]["start_gap"]) + 0.6 * level
            + self.prey_speed * t,
            (0.6 + 0.15 * level) * float(np.sin(self.prey_wander * t + self.prey_phase)),
            float(self.cfg["prey"]["height"]),
        ])
        self.data.mocap_pos[self.prey_mocap] = self.prey_pos
        self.prey_target = self.prey_pos.copy()
        self._prey_velocity[:] = (self.prey_pos - self._prev_prey_pos) / self.dt
        self._prev_prey_pos = self.prey_pos.copy()

    def _update_contacts(self):
        d = self.data
        self._contacts[:] = 0.0
        self._prey_contact = False
        self._jaw_force = 0.0
        self._prey_force = 0.0
        self._bad_ground_now = False
        n = d.ncon
        if n == 0:
            return
        g1 = d.contact.geom1[:n]
        g2 = d.contact.geom2[:n]
        fg, pg, jg = self.floor_geom, self.prey_geom, self.jaw_geom
        on_floor = (g1 == fg) | (g2 == fg)
        on_prey = (g1 == pg) | (g2 == pg)
        cf = self._cf
        for i in np.flatnonzero(on_floor | on_prey):
            i = int(i)
            a, b = int(g1[i]), int(g2[i])
            mujoco.mj_contactForce(self.model, d, i, cf)
            fn = abs(float(cf[0]))
            if on_floor[i]:
                hit_foot = False
                for j, g in enumerate(self.foot_geoms):
                    if (a == g or b == g) and fn > 0.1:
                        self._contacts[j] = 1.0
                        hit_foot = True
                if not hit_foot and (a in self.bad_ground_geoms or b in self.bad_ground_geoms):
                    self._bad_ground_now = True
            if on_prey[i]:
                self._prey_contact = True
                self._prey_force += fn
                if a == jg or b == jg:
                    self._jaw_force += fn

    # ------------------------------------------------------------------ #
    def command(self):
        t = self.elapsed
        if self.task == "loco":
            ph = self.phase_name()
            if ph == "accel":
                a = self.phase_t() / PHASE_DUR["loco"]["accel"]
                return float(SPEED_PROFILE["walk"] + a * (SPEED_PROFILE["run"] - SPEED_PROFILE["walk"])) * self.speed_scale, 0.0
            if ph == "decel":
                a = self.phase_t() / PHASE_DUR["loco"]["decel"]
                return float(SPEED_PROFILE["run"] * (1.0 - a)) * self.speed_scale, 0.0
            return float(SPEED_PROFILE[ph]) * self.speed_scale, 0.0
        if self.task == "chase":
            ph = self.phase_name()
            if ph in ("settle", "recover"):
                return 0.0, 0.0
            delta = self.prey_pos[:2] - self.data.xpos[self.torso_id, :2]
            dist = float(np.linalg.norm(delta))
            bearing = float(np.arctan2(delta[1], delta[0]))
            _, _, yaw = self.posture()
            err = float(np.arctan2(np.sin(bearing - yaw), np.cos(bearing - yaw)))
            yawrate = float(np.clip(err, -1.0, 1.0))
            if ph == "attack":
                return 0.0, yawrate
            # 上限取策略实际能达到的速度（阶段三实测峰值约 1.0 m/s）：
            # 把指令卡在 1.4 m/s 会让跟踪项长期饱和、失去梯度。
            return float(np.clip((dist - 0.45) / 0.9, 0.0, 1.0)
                         * min(1.0, 0.5 + 0.2 * self.curriculum)), yawrate
        return 0.0, 0.0

    def desired_mouth(self):
        if self.task == "reach":
            ph = self.phase_name()
            if ph == "settle":
                a = 0.0
            elif ph == "probe":
                a = _smoothstep(self.phase_t() / PHASE_DUR["reach"]["probe"])
            elif ph == "bite":
                a = 1.0
            elif ph == "retract":
                a = 1.0 - _smoothstep(self.phase_t() / PHASE_DUR["reach"]["retract"])
            else:
                a = 0.0
            return self.initial_mouth + a * (self.prey_target - self.initial_mouth)
        if self.task == "chase":
            return self.prey_pos
        return self.initial_mouth

    # ------------------------------------------------------------------ #
    def _obs(self):
        d = self.data
        R = d.xmat[self.torso_id].reshape(3, 3)
        torso_pos = d.xpos[self.torso_id]
        v = self._last_vel if self._last_vel is not None else self.torso_velocity()
        com = d.subtree_com[0]
        speed_cmd, yaw_cmd = self.command()
        parts = [
            d.qpos[self.qadr],
            d.qvel[self.vadr],
            R.T @ np.array([0.0, 0.0, -1.0]),
            v,
            [d.xpos[self.torso_id, 2]],
            self._contacts,
            [float(self._prey_contact), self._jaw_force * 0.02],
            R.T @ (self.prey_pos - torso_pos),
            R.T @ (v[3:] - self._prey_velocity),
            R.T @ (self.desired_mouth() - d.site_xpos[self.mouth_site]),
            [speed_cmd, yaw_cmd],
            R.T @ (com - torso_pos),
        ]
        if self.task in ("loco", "chase"):
            # 步态时钟：把"周期性"从需要探索的结构变成策略可直接使用的条件变量。
            # 这是一条**声明的**先验（预设了一个节律），报告里会明确标注它。
            ph = 2.0 * np.pi * self.gait_hz * self.elapsed
            parts.append(np.array([np.sin(ph), np.cos(ph)]))
        parts += [
            np.eye(PHASE_SLOTS)[self.phase_index()],
            np.eye(len(TASKS))[TASKS.index(self.task)],
            self.last_action,
        ]
        return np.concatenate(parts).astype(np.float32)

    # ------------------------------------------------------------------ #
    def step(self, action):
        action = np.clip(np.asarray(action, dtype=float), -1.0, 1.0)
        if action.shape != self.action_space.shape or not np.isfinite(action).all():
            raise ValueError("非法动作")

        self._move_prey()
        qpos, qvel, ctrl = self._qpos, self._qvel, self._ctrl
        bias = self._bias
        tau, tmp = self._tau, self._tmp
        kp, kd = self.kp, self.kd
        tl, sl = self.torque_limits, self.speed_limits
        gc_idx, gc_any = self.gc_idx, self.gc_any
        tail_driven = self.mode == "active"
        tail_mask = self.tail_mask
        targetq = np.clip(self.home + action * self.action_scale, self.q_lo, self.q_hi)

        e_step = 0.0
        peak = self.peak_torque
        sat = self.saturated
        mj_step = mujoco.mj_step
        model, d = self.model, self.data
        pushes = self.pushes
        push_end = self.push_end
        torso_id = self.torso_id
        xfrc = self._xfrc if pushes else None

        for _ in range(self.substeps):
            v = qvel[self.vadr]
            np.subtract(targetq, qpos[self.qadr], out=tau)
            tau *= kp
            np.multiply(v, kd, out=tmp)
            tau -= tmp
            if gc_any:
                tau[gc_idx] += bias[self.vadr][gc_idx]
            if tail_driven:
                sat += int(np.sum(np.abs(tau) >= tl))
            else:
                tau[tail_mask] = 0.0
            np.clip(tau, -tl, tl, out=tau)
            np.multiply(tau, v, out=tmp)
            tau[(np.abs(v) > sl) & (tmp > 0)] = 0.0
            pt = float(np.abs(tau).max())
            if pt > peak:
                peak = pt
            ctrl[:] = tau

            if xfrc is not None:
                xfrc[:] = 0.0
                t_now = d.time
                for t_push, force in pushes:
                    if t_push <= t_now < t_push + push_end:
                        xfrc[torso_id, 0] = force

            mj_step(model, d)
            np.multiply(tau, v, out=tmp)
            e_step += float(np.abs(tmp).sum()) * model.opt.timestep

        self.nctrl += (self.n if tail_driven else self.n - self.n_tail) * self.substeps
        self.saturated = sat
        self.peak_torque = peak
        self.elapsed = float(d.time)
        self.steps += 1
        self.energy += e_step
        prev_action = self.last_action
        self.last_action = action
        self._update_contacts()

        # ---------------- 运动学/动力学 ----------------
        pitch, roll, yaw = self.posture()
        vel = self.torso_velocity()
        self._last_vel = vel
        height = float(d.xpos[torso_id, 2])
        com = d.subtree_com[0]
        mouth = d.site_xpos[self.mouth_site]
        mouth_to_target = float(np.linalg.norm(mouth - self.prey_target))
        # 到目标【表面】的距离：嘴部最多只能贴到实体胶囊的表面，
        # 用球心距离会定出物理上不可达的阈值。
        mouth_to_surface = max(0.0, mouth_to_target - self.prey_radius)
        mouth_err = float(np.linalg.norm(mouth - self.desired_mouth()))
        prey_dist = float(np.linalg.norm(mouth - d.site_xpos[self.bite_site]))

        forward = float(np.cos(yaw) * vel[3] + np.sin(yaw) * vel[4])
        speed_cmd, yaw_cmd = self.command()
        track_err = abs(forward - speed_cmd) + abs(vel[2] - yaw_cmd)
        self._last_track_err = track_err
        self._track_sum += track_err
        self._track_n += 1
        # 转向表现：偏航跟踪误差 + 实际转过的角度（累计路径与净变化）
        self._yaw_err_sum += abs(float(vel[2]) - yaw_cmd)
        if self._prev_yaw is not None:
            d_yaw = float(np.arctan2(np.sin(yaw - self._prev_yaw), np.cos(yaw - self._prev_yaw)))
            self._yaw_travel += abs(d_yaw)
        self._prev_yaw = yaw
        ph = self.phase_name()

        # 滚动统计
        sway = float(np.linalg.norm(com[:2] - self._com0_xy))
        if sway > self._sway_max:
            self._sway_max = sway
        self._pitch_ss += pitch * pitch
        self._roll_ss += roll * roll
        tail_q_now = d.qpos[self.qadr[self.tail_idx]]
        np.minimum(self._tail_min, tail_q_now, out=self._tail_min)
        np.maximum(self._tail_max, tail_q_now, out=self._tail_max)
        dist = float(d.xpos[torso_id, 0] - self.initial_x)
        if abs(dist) > abs(self._dist_max):
            self._dist_max = dist
        if abs(forward) > self._speed_max:
            self._speed_max = float(abs(forward))

        # 足底打滑
        slip = 0.0
        if self._contacts[0] > 0:
            slip += self._foot_speed(self.foot_bodies[0])
        if self._contacts[1] > 0:
            slip += self._foot_speed(self.foot_bodies[1])
        self.slip_total += slip * self.dt

        # 步态事件（排除跌倒腾空，避免把摔倒误判成奔跑）
        tc = self.cfg["termination"]
        contact_any = bool(self._contacts[0] > 0 or self._contacts[1] > 0)
        airborne_ok = height > tc["min_height"] and abs(pitch) < tc["max_pitch"]
        if contact_any and self.airtime >= 0.04 and self.elapsed > 0.5 and airborne_ok:
            self.flight_events += 1
            if ph == "run":
                self.run_flight_events += 1
        if not contact_any:
            self.flight_steps += 1
            if ph == "run":
                self.run_flight_steps += 1
            self.airtime += self.dt
            if self.airtime > self.max_airtime:
                self.max_airtime = self.airtime
        else:
            self.airtime = 0.0
        n_sup = int(self._contacts[0] + self._contacts[1])
        if n_sup == 1:
            sup = 0 if self._contacts[0] > 0 else 1
            if self.last_single_support is not None and sup != self.last_single_support:
                self.support_switches += 1
            self.last_single_support = sup
        elif n_sup == 2:
            self.last_single_support = None
        # "额外迈步"：一脚离地 > 0.05 s 后重新落地才算一步（过滤接触抖动）
        for j in range(2):
            if self._contacts[j] > 0:
                if self._foot_air[j] > 0.05:
                    self.extra_steps += 1
                    # 步态塑形：落地时按摆动时长给分（目标 0.30 s）。
                    # 只奖励"迈步"这件事本身，不预设左右相位与频率。
                    if self.task == "loco" and speed_cmd > 0.05:
                        w = self.cfg["reward"]
                        self._air_bonus = float(w["airtime"]) * float(
                            np.exp(-((self._foot_air[j] - 0.30) / 0.20) ** 2))
                self._foot_air[j] = 0.0
            else:
                self._foot_air[j] += self.dt
        self.duty += self._contacts * self.dt

        # 尾巴角动量占比（每 2 个控制步算一次，够用且省算力）
        if self.steps % 2 == 0:
            l_tail, l_total = self._angular_momentum_split()
            self._mom_tail_sum += float(np.linalg.norm(l_tail))
            self._mom_total_sum += float(np.linalg.norm(l_total))
            self._mom_share_n += 1
        if self._prey_force > self.peak_prey_force:
            self.peak_prey_force = self._prey_force

        # ---------------- 事件 ----------------
        if self.task == "reach":
            if ph in ("probe", "bite") and mouth_to_surface < self.cfg["target"]["reach_surface_gap"]:
                self.reached = True
            # 简化咬合：嘴部抵住目标 + 主动闭颌 + 接触力在合理区间。
            # 不做"必须下颚几何体接触"的判定：头骨前端比下颚前缘更靠前，
            # 物体永远先撞到头骨，要求 jaw_geom 接触在几何上不可达。
            if (ph == "bite"
                    and mouth_to_surface < self.cfg["target"]["bite_surface_gap"]
                    and self._prey_contact
                    and d.qpos[self.qadr[self.jaw_idx]] < self.cfg["target"]["jaw_closed"]
                    and self.cfg["target"]["bite_force_min"] <= self._prey_force <= self.cfg["target"]["bite_force_max"]):
                self.bitten = True
        elif self.task == "chase":
            if self.attack_time is None and prey_dist < 0.55:
                self.attack_time = self.elapsed
            if (self.attack_time is not None and self._prey_contact
                    and d.qpos[self.qadr[self.jaw_idx]] < self.cfg["target"]["jaw_closed"]
                    and self.cfg["target"]["bite_force_min"] <= self._prey_force <= self.cfg["target"]["bite_force_max"]):
                self.bitten = True

        if self.perturb and self.pushes:
            ends = [t + self.push_end for t, _ in self.pushes if t + self.push_end <= self.elapsed]
            if ends:
                last_end = max(ends)
                if self.elapsed > last_end:
                    stable = self._is_stable(pitch, roll, vel)
                    self.recovery_hold = self.recovery_hold + self.dt if stable else 0.0
                    if self.recovery_time is None and self.recovery_hold >= 0.3:
                        self.recovery_time = max(0.0, self.elapsed - last_end - 0.3)

        if self.task == "reach" and ph == "rest":
            if self._rest_start is None:
                self._rest_start = self.elapsed
            if pitch < self._rest_pitch_min:
                self._rest_pitch_min = pitch
            if pitch > self._rest_pitch_max:
                self._rest_pitch_max = pitch
            if self.return_settle is None and self._is_stable(pitch, roll, vel):
                self.return_settle = self.elapsed - self._rest_start
            back = float(np.linalg.norm(mouth - self.initial_mouth))
            qd_tail = float(np.linalg.norm(d.qvel[self.vadr[self.tail_idx]]))
            if back < 0.14 and float(np.linalg.norm(vel[:3])) < 0.30 and qd_tail < 1.2:
                self.stood_again = True
                self.tail_rest_ok = True

        # ---------------- 终止 ----------------
        self.fallen = bool(
            height < tc["min_height"] or abs(pitch) > tc["max_pitch"] or abs(roll) > tc["max_roll"]
            or not np.isfinite(d.qpos).all() or self._bad_ground_now
        )

        reward, terms = self._reward(pitch, roll, height, vel, forward, speed_cmd,
                                     mouth_err, action, prev_action, e_step, slip, ph,
                                     prey_dist, mouth_to_surface)

        truncated = self.elapsed >= self.duration - 1e-9
        if self.trace is not None:
            tr = self.trace
            tr["t"].append(self.elapsed); tr["com_x"].append(float(com[0])); tr["com_y"].append(float(com[1]))
            tr["pitch"].append(pitch); tr["roll"].append(roll); tr["yaw"].append(yaw)
            tr["speed"].append(forward); tr["cmd_speed"].append(speed_cmd)
            tr["yaw_rate"].append(float(vel[2])); tr["cmd_yaw_rate"].append(yaw_cmd)
            tr["height"].append(height)
            tr["contact_l"].append(float(self._contacts[0])); tr["contact_r"].append(float(self._contacts[1]))
            tr["tail_q"].append(tail_q_now.tolist())
            tr["tail_qd"].append(d.qvel[self.vadr[self.tail_idx]].tolist())
            tr["prey_dist"].append(prey_dist); tr["mouth_err"].append(mouth_err)
            tr["reward"].append(float(reward))

        if self.fallen or truncated:
            info = self._full_info(pitch, roll, yaw, height, vel, forward, speed_cmd, yaw_cmd,
                                   mouth_err, mouth_to_target, prey_dist, terms, mouth_to_surface)
            info["success"] = bool(self._success(info)) if truncated else False
        else:
            info = {
                "task": self.task, "mode": self.mode, "phase": ph,
                "time": self.elapsed, "fallen": False,
                "reward_terms": terms,
            }
        return self._obs(), float(reward), bool(self.fallen), bool(truncated), info

    def _is_stable(self, pitch, roll, vel) -> bool:
        """恢复稳定的判据带：姿态偏差 + 角速度/线速度。

        阈值取 |pitch|<0.15 rad(8.6°)、|roll|<0.12 rad(6.9°)、|vel|<0.4。
        这是一个**定义**，报告里会写明；阈值过紧会让"已经站稳"被判成没恢复。
        """
        return (abs(pitch) < 0.15 and abs(roll) < 0.12
                and float(np.linalg.norm(vel[:3])) < 0.4)

    def _foot_speed(self, body_id) -> float:
        v = np.empty(6)
        mujoco.mj_objectVelocity(self.model, self.data, mujoco.mjtObj.mjOBJ_BODY, body_id, v, 0)
        return float(np.linalg.norm(v[3:5]))

    # ------------------------------------------------------------------ #
    def _angular_momentum_split(self):
        """尾巴 / 全身绕质心的角动量（世界系）。用 mj_subtreeVel，避免 Python 循环。"""
        d = self.data
        mujoco.mj_subtreeVel(self.model, d)
        com = d.subtree_com[0]
        m_tot = self.total_mass
        v_tot = d.subtree_linvel[0] / m_tot
        # cross(com, v) 手写，避免 np.cross 在 3 向量上的高开销
        cx, cy, cz = float(com[0]), float(com[1]), float(com[2])
        l_total = d.subtree_angmom[0] - m_tot * np.array([
            cy * v_tot[2] - cz * v_tot[1],
            cz * v_tot[0] - cx * v_tot[2],
            cx * v_tot[1] - cy * v_tot[0],
        ])
        ids = self.tail_body_ids
        masses = self.tail_body_mass
        v = d.subtree_linvel[ids] / masses[:, None]
        am = d.subtree_angmom[ids]
        l_tail = np.array([
            float(np.sum(am[:, 0] - masses * (cy * v[:, 2] - cz * v[:, 1]))),
            float(np.sum(am[:, 1] - masses * (cz * v[:, 0] - cx * v[:, 2]))),
            float(np.sum(am[:, 2] - masses * (cx * v[:, 1] - cy * v[:, 0]))),
        ])
        return l_tail, l_total

    # ------------------------------------------------------------------ #
    def _reward(self, pitch, roll, height, vel, forward, speed_cmd, mouth_err,
                action, prev_action, e_step, slip, ph, prey_dist=0.0, mouth_to_surface=0.0):
        w = self.cfg["reward"]
        tgt = self.cfg["target"]
        terms = {
            "alive": w["alive"],
            "upright": w["upright"] * float(np.exp(-4.0 * (pitch * pitch + roll * roll))),
            "height": w["height"] * float(np.exp(-30.0 * (height - self.cfg["robot"]["stand_height"]) ** 2)),
        }
        task_r = 0.0
        d = self.data
        if self.task == "stand":
            com = d.subtree_com[0]
            task_r = float(np.exp(-8.0 * float(np.sum((com[:2] - self._com0_xy) ** 2)) - 2.0 * float(np.sum(vel ** 2))))
        elif self.task == "reach":
            # 奖励量用【到目标表面的距离】而不是到球心：球心在实体内部不可达。
            # 核宽用 reach_kernel（0.15 m），比原先的 exp(-150·x²)（等效 σ≈0.058）
            # 宽得多——否则机器人在 0.2 m 外时奖励只有 0.002，几乎没有梯度可学。
            k = float(self.cfg["target"]["reach_kernel"])
            jaw_q = float(d.qpos[self.qadr[self.jaw_idx]])
            if ph in ("settle", "probe", "bite"):
                task_r = float(np.exp(-(mouth_to_surface / k) ** 2))
            else:  # retract：奖励嘴部回到起始位
                task_r = float(np.exp(-(mouth_err / k) ** 2))
            if ph in ("probe", "bite"):
                # 稠密塑形：靠近目标的同时合颌才给分（离得远时该项为 0，防"一直闭嘴"投机）
                task_r += w["bite_prox"] * float(
                    np.exp(-(mouth_to_surface / k) ** 2)) * float(np.exp(-20.0 * jaw_q ** 2))
            if ph == "bite":
                task_r += w["bite_close"] * float(np.exp(-20.0 * jaw_q ** 2))
                if self.bitten:
                    task_r += 1.0
            elif ph == "rest":
                com = d.subtree_com[0]
                task_r = 0.6 * float(np.exp(-8.0 * float(np.sum((com[:2] - self._com0_xy) ** 2)))) \
                    + 0.4 * float(np.exp(-20.0 * float(np.sum(vel ** 2))))
        elif self.task == "loco":
            task_r = float(np.exp(-2.5 * (forward - speed_cmd) ** 2 - 2.0 * float(vel[4] ** 2) - 2.0 * float(vel[2] ** 2)))
            # 前进进度项：静止时为 0，速度达到指令即饱和。
            # 必须用 **对称** 截断 clip(v/c, -1, 1)：
            # 若用单边 clip(v/c, 0, 1)，零均值的探索噪声会让它取正期望
            # （E[max(v,0)] > 0），策略靠抖动就能骗到奖励，
            # 训练期指标虚高而确定性执行原地不动。
            # 对称截断下噪声期望为 0，只有持续前进才拿得到分；
            # 后退会被扣分（这是对的）。
            if speed_cmd > 0.05 and ph in ("walk", "accel", "run"):
                task_r += w["progress"] * float(np.clip(forward / speed_cmd, -1.0, 1.0))
            if ph == "run" and self.airtime > 0.0:
                task_r += 0.25 * float(np.exp(-3.0 * (forward - speed_cmd) ** 2))
            # 落地时的摆动时长塑形（一次性奖励，在 step 里置位）
            task_r += self._air_bonus
            self._air_bonus = 0.0
        elif self.task == "chase":
            task_r = float(np.exp(-1.5 * prey_dist ** 2))
            jaw_q = float(d.qpos[self.qadr[self.jaw_idx]])
            # 追逐段必须同时奖励"跑起来"：只看距离的话，策略会把
            # 之前学到的奔跑步态训弱（实测均速从 1.0 掉到 0.33 m/s，追不上猎物）。
            if ph == "chase" and speed_cmd > 0.05:
                task_r += 0.6 * float(np.exp(-2.5 * (forward - speed_cmd) ** 2))
                task_r += w["progress"] * float(np.clip(forward / speed_cmd, -1.0, 1.0))
            if self.bitten:
                task_r += 1.5
            if ph == "attack":
                task_r += 0.3 * float(np.exp(-20.0 * forward ** 2))
                task_r += w["bite_close"] * float(np.exp(-1.5 * prey_dist ** 2)) * float(
                    np.exp(-20.0 * jaw_q ** 2))
        terms["task"] = w["task"] * task_r

        if self.task in ("reach", "chase") and ph not in ("bite", "attack"):
            jaw_q = float(d.qpos[self.qadr[self.jaw_idx]])
            terms["jaw_open"] = w["jaw_open"] * float(np.exp(-20.0 * (jaw_q - 0.25) ** 2))

        terms["energy"] = -w["energy"] * e_step / self.dt
        terms["action_rate"] = -w["action_rate"] * float(np.mean((action - prev_action) ** 2))
        terms["slip"] = -w["slip"] * slip
        terms["target_force"] = -w["target_force"] * max(0.0, self._prey_force - tgt["safe_force"])
        if self.fallen:
            terms["fall"] = -w["fall"]
        return float(sum(terms.values())), terms

    def _success(self, info) -> bool:
        if self.fallen:
            return False
        if self.task == "stand":
            # 成功 = 没倒 + 晃动受限 +（有外力时）确实恢复了稳定。
            # "额外迈步" 只作为**测量指标**报告（越少越好），不作为成功门槛：
            # 受扰后迈一步恢复是正确行为，不该被判失败。
            ok = self._sway_max < 0.30
            if self.perturb:
                return bool(ok and self.recovery_time is not None)
            return bool(ok)
        if self.task == "reach":
            return bool(self.reached and self.bitten and self.stood_again and self.tail_rest_ok
                        and self.return_settle is not None)
        if self.task == "loco":
            return bool(info["distance"] > 1.5 and info["mean_track_err"] < 0.45
                        and abs(info["speed"]) < 0.30 and info["run_flight_events"] >= 2)
        if self.task == "chase":
            return bool(self.attack_time is not None and self.bitten)
        return False

    # ------------------------------------------------------------------ #
    def _full_info(self, pitch, roll, yaw, height, vel, forward, speed_cmd, yaw_cmd,
                   mouth_err, mouth_to_target, prey_dist, terms, mouth_to_surface):
        steps = max(1, self.steps)
        n = max(1, self._mom_share_n)
        rom = np.degrees(self._tail_max - self._tail_min)
        rom = np.where(np.isfinite(rom), rom, 0.0)
        info = {
            "task": self.task, "mode": self.mode, "phase": self.phase_name(),
            "phase_index": self.phase_index(), "time": self.elapsed,
            "pitch": pitch, "roll": roll, "yaw": yaw, "height": height,
            "vx": float(vel[3]), "vy": float(vel[4]), "speed": forward,
            "speed_cmd": speed_cmd, "yaw_rate": float(vel[2]), "yaw_cmd": yaw_cmd,
            "track_err": self._last_track_err,
            "mean_track_err": self._track_sum / max(1, self._track_n),
            "mean_yaw_track_err": self._yaw_err_sum / max(1, self._track_n),
            "yaw_travel_deg": float(np.degrees(self._yaw_travel)),
            "yaw_net_deg": float(np.degrees(yaw)),
            "distance": self._dist_max,
            "speed_max": self._speed_max,
            "com_sway": self._sway_max,
            "pitch_rms": float(np.sqrt(self._pitch_ss / steps)),
            "roll_rms": float(np.sqrt(self._roll_ss / steps)),
            "mouth_err": mouth_err, "mouth_to_target": mouth_to_target, "prey_dist": prey_dist,
            "mouth_to_surface": mouth_to_surface,
            "prey_force": self._prey_force, "jaw_force": self._jaw_force,
            "prey_contact": bool(self._prey_contact),
            "foot_left": float(self._contacts[0]), "foot_right": float(self._contacts[1]),
            "airtime": self.airtime, "max_airtime": self.max_airtime,
            "flight_events": self.flight_events, "run_flight_events": self.run_flight_events,
            "flight_steps": self.flight_steps, "flight_ratio": self.flight_steps / steps,
            "run_flight_ratio": self.run_flight_steps / steps,
            "support_switches": self.support_switches,
            "extra_steps": self.extra_steps,
            "duty_left": float(self.duty[0] / (self.elapsed + 1e-9)),
            "duty_right": float(self.duty[1] / (self.elapsed + 1e-9)),
            "gait": self._gait_label(),
            "energy_j": self.energy, "slip_m": self.slip_total,
            "peak_torque_nm": self.peak_torque, "peak_prey_force_n": self.peak_prey_force,
            "saturation_fraction": self.saturated / max(1, self.nctrl),
            "recovery_s": self.recovery_time,
            "return_sway_rad": float(self._rest_pitch_max - self._rest_pitch_min)
            if self._rest_pitch_max > self._rest_pitch_min else None,
            "return_settle_s": self.return_settle,
            "attack_time": self.attack_time,
            "tail_rom_deg": rom.tolist(),
            "tail_momentum_share": self._mom_tail_sum / max(1e-6, self._mom_total_sum),
            "tail_momentum_abs": self._mom_tail_sum / n,
            "total_momentum_abs": self._mom_total_sum / n,
            "tail_q": self.data.qpos[self.qadr[self.tail_idx]].tolist(),
            "tail_qd": self.data.qvel[self.vadr[self.tail_idx]].tolist(),
            "reached": bool(self.reached), "bitten": bool(self.bitten),
            "stood_again": bool(self.stood_again), "tail_rest_ok": bool(self.tail_rest_ok),
            "fallen": bool(self.fallen),
            "episode_end": True,
            "reward_terms": terms,
        }
        if self.task == "loco":
            # 分开报告：会移动并停住（运动学层面） vs 真的出现腾空相（奔跑证据）
            moving = (not self.fallen and abs(self._dist_max) > 1.5
                      and info["mean_track_err"] < 0.45 and abs(forward) < 0.30)
            info["success_locomotion"] = bool(moving)
            info["success_run"] = bool(moving and self.run_flight_events >= 2)
        if self.task == "chase":
            info["success_approach"] = bool(
                not self.fallen and self.attack_time is not None and prey_dist < 0.55)
        return info

    def _gait_label(self) -> str:
        if self.elapsed < 0.5:
            return "startup"
        if self.fallen:
            return "fallen"
        if self.flight_events >= 2 and self.max_airtime > 0.05:
            return "run"
        if self.support_switches >= 4 and self.flight_steps == 0:
            return "walk"
        if self.support_switches >= 2:
            return "stepping"
        return "standing"

    # ------------------------------------------------------------------ #
    def state(self) -> dict:
        """给可视化用的完整状态快照。"""
        pitch, roll, yaw = self.posture()
        vel = self.torso_velocity()
        com = self.data.subtree_com[0]
        return {
            "time": self.elapsed, "phase": self.phase_name(), "task": self.task, "mode": self.mode,
            "pitch": pitch, "roll": roll, "yaw": yaw,
            "height": float(self.data.xpos[self.torso_id, 2]),
            "com": com.tolist(), "vel": vel.tolist(),
            "speed": float(np.cos(yaw) * vel[3] + np.sin(yaw) * vel[4]),
            "foot": self._contacts.tolist(),
            "prey": self.prey_pos.tolist(),
            "tail_q": self.data.qpos[self.qadr[self.tail_idx]].tolist(),
            "tail_qd": self.data.qvel[self.vadr[self.tail_idx]].tolist(),
            "jaw": float(self.data.qpos[self.qadr[self.jaw_idx]]),
            "gait": self._gait_label(),
            "fallen": bool(self.fallen),
            "contacts": [
                [float(self.data.contact[i].pos[j]) for j in range(3)]
                for i in range(self.data.ncon)
            ],
        }

    def render(self):
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=540, width=960)
            self.camera = mujoco.MjvCamera()
            self.camera.azimuth = 135.0 if self.dimension == "3d" else 90.0
            self.camera.elevation = -14.0
            self.camera.distance = 3.4
        self.camera.lookat[:] = self.data.xpos[self.torso_id] + np.array([-0.1, 0.0, 0.02])
        opt = mujoco.MjvOption()
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
        opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
        self.renderer.update_scene(self.data, camera=self.camera, scene_option=opt)
        return self.renderer.render()

    def close(self):
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
