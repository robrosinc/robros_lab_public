"""IGRIS-C locomotion asset configuration."""

from pathlib import Path

import isaaclab.sim as sim_utils
from isaaclab.actuators import DelayedPDActuatorCfg
from isaaclab.assets.articulation import ArticulationCfg

_ROBROS_ASSET_DIR = Path(__file__).resolve().parent / "robros"
_USD_SUBDIR = Path("from_urdf") / "usd"


def _robot_usd_dir(robot_name: str, variant_dir: str | None = None) -> Path:
    usd_dir = _ROBROS_ASSET_DIR / robot_name / _USD_SUBDIR
    if variant_dir is not None:
        usd_dir = usd_dir / variant_dir
    return usd_dir


def _usd_path(robot_name: str, variant_dir: str | None = None) -> str:
    usd_path = _robot_usd_dir(robot_name, variant_dir) / f"{robot_name}.usd"
    if not usd_path.is_file():
        raise FileNotFoundError(f"Robot USD does not exist: {usd_path}")
    return str(usd_path)


IGRISC_LOCOMOTION_ACTUATOR_CFG = {
    "X12-150": DelayedPDActuatorCfg(
        joint_names_expr=[".*Hip_Pitch.*", ".*Knee.*"],
        velocity_limit=10.4720,
        velocity_limit_sim=10.4720,
        effort_limit=150.0,
        effort_limit_sim=150.0,
        stiffness={".*.*": 118.435253},
        damping={".*.*": 15.079645},
        armature={".*Hip_Pitch.*": 0.3, ".*Knee.*": 0.18},
        friction={".*Hip_Pitch.*": 0.5, ".*Knee.*": 0.5},
        viscous_friction={".*Hip_Pitch.*": 2.4, ".*Knee.*": 2.2},
        min_delay=0,
        max_delay=2,
    ),
    "X8-120": DelayedPDActuatorCfg(
        joint_names_expr=[".*Hip_Roll.*"],
        velocity_limit=13.2994,
        velocity_limit_sim=13.2994,
        effort_limit=120.0,
        effort_limit_sim=120.0,
        stiffness=122.136354,
        damping=10.367256,
        armature=0.06,
        friction=0.1,
        viscous_friction=3.0,
        min_delay=0,
        max_delay=2,
    ),
    "X6-60": DelayedPDActuatorCfg(
        joint_names_expr=[".*Hip_Yaw.*", ".*Waist_Yaw.*", ".*_Shoulder_.*", ".*_Elbow_.*"],
        velocity_limit=16.0221,
        velocity_limit_sim=16.0221,
        effort_limit=60.0,
        effort_limit_sim=60.0,
        stiffness={".*.*": 121.514570},
        damping={".*.*": 6.077864},
        armature={".*.*": 0.02},
        friction={".*.*": 1.01},
        viscous_friction={".*.*": 1.8},
        min_delay=0,
        max_delay=2,
    ),
    "PLA": DelayedPDActuatorCfg(
        joint_names_expr=[".*Ankle.*", ".*Waist_Roll.*", ".*Waist_Pitch.*"],
        velocity_limit={
            ".*Waist_Roll.*": 8.6608,
            ".*Waist_Pitch.*": 15.6655,
            ".*Ankle_Pitch.*": 18.1383,
            ".*Ankle_Roll.*": 19.8934,
        },
        velocity_limit_sim={
            ".*Waist_Roll.*": 8.6608,
            ".*Waist_Pitch.*": 15.6655,
            ".*Ankle_Pitch.*": 18.1383,
            ".*Ankle_Roll.*": 19.8934,
        },
        effort_limit={
            ".*Waist_Roll.*": 217.0253,
            ".*Waist_Pitch.*": 120.0,
            ".*Ankle_Pitch.*": 163.1163,
            ".*Ankle_Roll.*": 148.7251,
        },
        effort_limit_sim={
            ".*Waist_Roll.*": 217.0253,
            ".*Waist_Pitch.*": 120.0,
            ".*Ankle_Pitch.*": 163.1163,
            ".*Ankle_Roll.*": 148.7251,
        },
        stiffness={
            ".*Waist_Roll.*": 660.634418,
            ".*Waist_Pitch.*": 230.579719,
            ".*Ankle_Pitch.*": 93.373568,
            ".*Ankle_Roll.*": 77.634366,
        },
        damping={
            ".*Waist_Roll.*": 34.67,
            ".*Waist_Pitch.*": 12.10,
            ".*Ankle_Pitch.*": 14.863025,
            ".*Ankle_Roll.*": 12.355893,
        },
        armature={
            ".*Waist_Roll.*": 0.114,
            ".*Waist_Pitch.*": 0.047,
            ".*Ankle_Pitch.*": 0.14,
            ".*Ankle_Roll.*": 0.12,
        },
        friction={
            ".*Waist_Roll.*": 3.6,
            ".*Waist_Pitch.*": 1.2,
            ".*Ankle_Pitch.*": 0.5,
            ".*Ankle_Roll.*": 0.5,
        },
        viscous_friction={
            ".*Waist_Roll.*": 7.5,
            ".*Waist_Pitch.*": 2.5,
            ".*Ankle_Pitch.*": 0.1,
            ".*Ankle_Roll.*": 0.1,
        },
        min_delay=0,
        max_delay=2,
    ),
}


IGRIS_C_LOCOMOTION_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_usd_path("igris_c", "extras/locomotion"),
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),
        joint_pos={
            ".*_Hip_Pitch_.*": -0.2,
            ".*_Hip_Roll_.*": 0.0,
            ".*_Hip_Yaw_.*": 0.0,
            ".*_Knee_Pitch_.*": 0.3,
            ".*_Ankle_Pitch_.*": -0.1,
            ".*_Ankle_Roll_.*": 0.0,
            ".*_Waist_.*": 0.0,
            ".*_Shoulder_Pitch_Left": 0.13,
            ".*_Shoulder_Roll_Left": 0.13,
            ".*_Shoulder_Yaw_Left": 0.0,
            ".*_Elbow_Pitch_Left": -0.3,
            ".*_Shoulder_Pitch_Right": 0.13,
            ".*_Shoulder_Roll_Right": -0.13,
            ".*_Shoulder_Yaw_Right": -0.0,
            ".*_Elbow_Pitch_Right": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    actuators=IGRISC_LOCOMOTION_ACTUATOR_CFG,
    soft_joint_pos_limit_factor=0.9,
)


__all__ = ["IGRIS_C_LOCOMOTION_CFG"]
