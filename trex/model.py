"""MuJoCo 机器霸王龙模型构建。

设计要点（与尾巴"活性"直接相关，改动前请先读 config.json 的 assumptions）：

1. 尾巴结构完全由 config.json 的 `tail` 段决定：段数、长度、质量分布、
   关节轴、活动范围、力矩/速度上限、被动阻尼。改配置即可换尾巴。
2. 尾关节的被动 `stiffness` 默认为 0：不预置"回中弹簧"。尾巴的垂坠与
   跟随完全来自重力与惯性，而不是脚本化的摆动。
3. 三种对照（active / passive / fixed）使用**同一套形态与被动参数**：
   - active ：尾关节由策略执行器驱动
   - passive：尾关节执行器力矩恒为 0（自由被动摆动）
   - fixed  ：用 equality 把尾关节刚性锁死（尾巴＝躯干刚体延伸）
   三者的 XML 除 fixed 多出的 equality 约束外逐字相同。
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = ROOT / "config.json"

DIMENSIONS = ("planar", "3d")
MODES = ("active", "passive", "fixed")


# --------------------------------------------------------------------------- #
# 配置
# --------------------------------------------------------------------------- #
def _deep_merge(base: dict, overlay: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (overlay or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


def load_config(path=None, overlay=None) -> dict:
    """读取主配置，可选叠加一个 overlay（dict 或 json 路径）。"""
    cfg = json.loads(Path(path or CONFIG_PATH).read_text(encoding="utf-8"))
    if overlay is None:
        return cfg
    if isinstance(overlay, (str, Path)):
        overlay = json.loads(Path(overlay).read_text(encoding="utf-8"))
    return _deep_merge(cfg, overlay)


def tail_profile(cfg: dict):
    """返回尾巴的质量/半径/力矩/速度分布，便于外部核对与复用。"""
    t = cfg["tail"]
    n = int(t["segments"])
    ratio = float(t["segment_mass_ratio"])
    raw = np.array([ratio ** i for i in range(n)], dtype=float)
    masses = float(t["mass"]) * raw / raw.sum()
    if n == 1:
        radii = np.array([float(t["base_radius"])])
    else:
        tip = float(t["tip_radius_ratio"])
        frac = 1.0 - np.arange(n, dtype=float) / (n - 1)
        radii = float(t["base_radius"]) * (frac * (1.0 - tip) + tip)
    if n == 1:
        torques = np.array([float(t["torque_proximal"])])
        speeds = np.array([float(t["speed_proximal"])])
    else:
        frac = np.arange(n, dtype=float) / (n - 1)
        torques = float(t["torque_proximal"]) + frac * (
            float(t["torque_distal"]) - float(t["torque_proximal"])
        )
        speeds = float(t["speed_proximal"]) + frac * (
            float(t["speed_distal"]) - float(t["speed_proximal"])
        )
    return {
        "segments": n,
        "length": float(t["length"]),
        "segment_length": float(t["length"]) / n,
        "masses": masses,
        "radii": radii,
        "torques": torques,
        "speeds": speeds,
    }


# --------------------------------------------------------------------------- #
# XML 构建
# --------------------------------------------------------------------------- #
class _Builder:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.names: list[str] = []      # 执行器关节顺序（不含根关节）
        self.torques: list[float] = []
        self.speeds: list[float] = []
        self.motors: list[str] = []

    def joint(
        self,
        name: str,
        *,
        axis: str = "0 1 0",
        bounds=(-65.0, 65.0),
        torque: float = 55.0,
        speed: float = 7.0,
        extra: str = "",
    ) -> str:
        self.names.append(name)
        self.torques.append(float(torque))
        self.speeds.append(float(speed))
        self.motors.append(
            f'<motor name="{name}_motor" joint="{name}" '
            f'ctrlrange="{-torque} {torque}"/>'
        )
        return (
            f'<joint name="{name}" axis="{axis}" '
            f'range="{bounds[0]} {bounds[1]}" {extra}/>'.replace("  ", " ")
        )


def build_xml(cfg: dict, dimension: str = "planar", mode: str = "active"):
    """构建模型 XML，并返回 (xml, meta)。"""
    if dimension not in DIMENSIONS:
        raise ValueError(f"dimension 必须是 {DIMENSIONS}")
    if mode not in MODES:
        raise ValueError(f"mode 必须是 {MODES}")

    r, c, t = cfg["robot"], cfg["control"], cfg["tail"]
    lim = cfg["joint_limits"]
    tail = tail_profile(cfg)
    b = _Builder(cfg)

    if dimension == "3d":
        root = '<freejoint name="root"/>'
    else:
        # 声明的实验限制：矢状面内的平面浮动基座
        root = (
            '<joint name="root_x" type="slide" axis="1 0 0" limited="false" '
            'damping="0" armature="0"/>'
            '<joint name="root_z" type="slide" axis="0 0 1" limited="false" '
            'damping="0" armature="0"/>'
            '<joint name="root_pitch" axis="0 1 0" limited="false" '
            'damping="0" armature="0"/>'
        )

    # ---------------- 腿 ----------------
    legs = ""
    for side, y in (("left", r["hip_y_offset"]), ("right", -r["hip_y_offset"])):
        pre = ""
        if dimension == "3d":
            pre += b.joint(
                f"{side}_hip_roll", axis="1 0 0", bounds=lim["hip_roll"],
                torque=c["leg_torque"], speed=c["leg_speed"],
            )
            pre += b.joint(
                f"{side}_hip_yaw", axis="0 0 1", bounds=lim["hip_yaw"],
                torque=c["leg_torque"], speed=c["leg_speed"],
            )
        pre += b.joint(
            f"{side}_hip", bounds=lim["hip"], torque=c["leg_torque"], speed=c["leg_speed"]
        )
        knee = b.joint(
            f"{side}_knee", bounds=lim["knee"], torque=c["leg_torque"], speed=c["leg_speed"]
        )
        ankle = b.joint(
            f"{side}_ankle", bounds=lim["ankle"], torque=c["leg_torque"], speed=c["leg_speed"]
        )
        thigh = r["thigh_length"]
        shin = r["shin_length"]
        fs = r["foot_size"]
        legs += f'''<body name="{side}_thigh" pos="0 {y} 0">{pre}
      <geom name="{side}_thigh_geom" type="capsule" fromto="0 0 0 {thigh*0.35:.4f} 0 {-thigh:.4f}" size="{r['thigh_radius']}" mass="{r['thigh_mass']}" rgba=".23 .43 .38 1"/>
      <body name="{side}_shin" pos="{thigh*0.35:.4f} 0 {-thigh:.4f}">{knee}
        <geom name="{side}_shin_geom" type="capsule" fromto="0 0 0 {-thigh*0.35:.4f} 0 {-shin:.4f}" size="{r['shin_radius']}" mass="{r['shin_mass']}" rgba=".20 .38 .34 1"/>
        <body name="{side}_foot" pos="{-thigh*0.35:.4f} 0 {-shin:.4f}">{ankle}
          <geom name="{side}_foot" type="box" pos="{fs[0]*0.34:.4f} 0 {-fs[2]:.4f}" size="{fs[0]} {fs[1]} {fs[2]}" mass="{r['foot_mass']}" friction="{cfg['physics']['friction']*1.2:.3f} 0.005 0.0001" rgba=".13 .22 .22 1"/>
          <site name="{side}_ankle_site" pos="0 0 0" size=".012" rgba=".9 .9 .2 1"/>
        </body>
      </body>
    </body>'''

    # ---------------- 颈 / 头 / 颌 ----------------
    neck = b.joint("neck", bounds=lim["neck"], torque=c["neck_torque"], speed=c["neck_speed"])
    # 下颌是轻阻尼快速机构：显式覆盖默认关节阻尼，否则 kd+damping 会把闭颌速度压到
    # ~1 rad/s，策略一步只能转 0.02 rad，永远来不及在咬合窗口内闭嘴。
    jaw = b.joint(
        "jaw", bounds=lim["jaw"], torque=c["jaw_torque"], speed=c["jaw_speed"],
        extra=f'damping="{c.get("jaw_damping", 0.05)}"',
    )
    head = f'''<body name="neck" pos=".20 0 .08">{neck}
      <geom name="neck_geom" type="capsule" fromto="0 0 0 .18 0 .14" size=".075" mass="{r['neck_mass']}" rgba=".22 .44 .36 1"/>
      <body name="head" pos=".18 0 .14">
        <geom name="skull" type="box" pos=".12 0 .035" size=".17 .10 .075" mass="{r['head_mass']}" rgba=".26 .50 .40 1"/>
        <geom name="eye_l" type="sphere" pos=".10 -.098 .10" size=".021" mass=".005" contype="0" conaffinity="0" rgba="1 .7 .15 1"/>
        <geom name="eye_r" type="sphere" pos=".10 .098 .10" size=".021" mass=".005" contype="0" conaffinity="0" rgba="1 .7 .15 1"/>
        <site name="mouth" pos=".27 0 -.045" size=".014" rgba="1 .4 .1 1"/>
        <body name="jaw" pos="-.01 0 -.055">{jaw}
          <geom name="jaw_geom" type="box" pos=".14 0 0" size=".15 .082 .025" mass="{r['jaw_mass']}" rgba=".30 .52 .43 1"/>
        </body>
      </body>
    </body>'''

    # ---------------- 尾巴（研究的自变量） ----------------
    seg = tail["segment_length"]
    tail_xml = ""
    closes = ""
    equality = ""
    for i in range(tail["segments"]):
        extra = f'stiffness="{t["stiffness"]}" damping="{t["damping"]}"'
        j = b.joint(
            f"tail_{i}_pitch", bounds=(-t["range_deg"], t["range_deg"]),
            torque=float(tail["torques"][i]), speed=float(tail["speeds"][i]), extra=extra,
        )
        if dimension == "3d":
            j += b.joint(
                f"tail_{i}_yaw", axis="0 0 1", bounds=(-t["yaw_range_deg"], t["yaw_range_deg"]),
                torque=float(tail["torques"][i]), speed=float(tail["speeds"][i]), extra=extra,
            )
        pos = "-.24 0 .04" if i == 0 else f"{-seg:.5f} 0 0"
        tail_xml += (
            f'<body name="tail_{i}" pos="{pos}">{j}'
            f'<geom name="tail_{i}_geom" type="capsule" fromto="0 0 0 {-seg:.5f} 0 0" '
            f'size="{tail["radii"][i]:.5f}" mass="{tail["masses"][i]:.5f}" '
            f'rgba=".22 .48 .42 1"/>'
        )
        closes += "</body>"
    if mode == "fixed":
        locked = [n for n in b.names if n.startswith("tail_")]
        equality = "".join(
            f'<joint joint1="{n}" polycoef="0 0 0 0 0" solref=".004 1"/>' for n in locked
        )

    # ---------------- 目标 / 猎物（mocap） ----------------
    prey = (
        '<body name="prey" mocap="true" pos="1.0 0 0.95">'
        f'<geom name="prey_geom" type="capsule" fromto="0 0 -{cfg["target"]["radius"]:.4f} 0 0 {cfg["target"]["radius"]:.4f}" '
        f'size="{cfg["target"]["radius"]}" '
        'mass=".001" contype="1" conaffinity="1" rgba=".95 .35 .12 .85"/>'
        '<site name="bite_point" pos="0 0 0" size=".02" rgba="1 .9 .2 1"/>'
        "</body>"
    )

    p = cfg["physics"]
    xml = f'''<mujoco model="trex_{dimension}_{mode}">
  <compiler angle="degree" autolimits="true"/>
  <option timestep="{p['timestep']}" integrator="{p['integrator']}" gravity="0 0 -9.81" iterations="{p['solver_iterations']}"/>
  <default>
    <joint damping="1.2" armature=".025"/>
    <geom friction="{p['friction']} .005 .0001" condim="3" solref=".008 1" rgba=".27 .52 .43 1"/>
    <site rgba="1 .5 0 1"/>
    <motor ctrllimited="true"/>
  </default>
  <visual>
    <global offwidth="1280" offheight="720"/>
    <headlight diffuse=".7 .7 .7" ambient=".4 .4 .4"/>
    <map shadowclip="2.0"/>
    <quality shadowsize="2048"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" rgb1=".16 .19 .24" rgb2=".20 .24 .29" width="256" height="256"/>
    <material name="ground" texture="grid" texrepeat="10 10" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="0 -3 5" dir="0 0 -1"/>
    <geom name="floor" type="plane" size="40 40 .1" material="ground"/>
    {prey}
    <body name="torso" pos="0 0 {r['stand_height']}">{root}
      <geom name="torso" type="ellipsoid" size="{r['torso_size'][0]} {r['torso_size'][1]} {r['torso_size'][2]}" mass="{r['torso_mass']}" rgba=".27 .52 .43 1"/>
      <site name="torso_site" pos="0 0 0" size=".02" rgba="1 .2 .2 1"/>
      {legs}
      {head}
      {tail_xml}{closes}
    </body>
  </worldbody>
  <equality>{equality}</equality>
  <actuator>{''.join(b.motors)}</actuator>
  <contact>
    <exclude body1="head" body2="jaw"/>
  </contact>
</mujoco>'''

    meta = {
        "dimension": dimension,
        "mode": mode,
        "names": list(b.names),
        "torques": np.array(b.torques, dtype=float),
        "speeds": np.array(b.speeds, dtype=float),
        "tail": tail,
    }
    return xml, meta


def load_model(cfg: dict, dimension: str = "planar", mode: str = "active"):
    import mujoco

    xml, meta = build_xml(cfg, dimension, mode)
    model = mujoco.MjModel.from_xml_string(xml)
    return model, meta, xml


def export_models(cfg: dict | None = None, folder=None) -> list[Path]:
    """导出全部 6 个 XML 及惯量快照，用于核对对照实验形态一致性。"""
    import mujoco

    cfg = cfg or load_config()
    folder = Path(folder or ROOT / "models")
    folder.mkdir(parents=True, exist_ok=True)
    written = []
    for dim in DIMENSIONS:
        for mode in MODES:
            xml, _ = build_xml(cfg, dim, mode)
            model = mujoco.MjModel.from_xml_string(xml)
            path = folder / f"{dim}_{mode}.xml"
            path.write_text(xml, encoding="utf-8")
            written.append(path)
            snapshot = {
                "dimension": dim,
                "mode": mode,
                "nbody": int(model.nbody),
                "nq": int(model.nq),
                "nv": int(model.nv),
                "nu": int(model.nu),
                "total_mass": float(model.body_mass.sum()),
                "body_names": [
                    mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, i)
                    for i in range(model.nbody)
                ],
                "body_mass": model.body_mass.tolist(),
                "body_inertia": model.body_inertia.tolist(),
            }
            (folder / f"{dim}_{mode}_inertia.json").write_text(
                json.dumps(snapshot, indent=2), encoding="utf-8"
            )
    return written


if __name__ == "__main__":
    for path in export_models():
        print(path)
