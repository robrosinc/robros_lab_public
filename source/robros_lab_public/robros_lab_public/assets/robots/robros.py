"""Configuration for ROBROS robots.

The following configurations are available:

* :obj:`IGRIS_C_LOCOMOTION_CFG`: IGRIS-C humanoid robot

"""

import math
from pathlib import Path
from typing import Literal

import isaaclab.sim as sim_utils
from isaaclab.assets.articulation import ArticulationCfg

from robros_lab_public.actuators import DelayedDCMotorWithSlackCfg

##
# Robot interface schema.
##

# Canonical IGRIS-C joint-vector order used at policy, dataset, checkpoint, and
# deployment boundaries.  Physics backends may expose a different native order;
# resolve these names to backend indices instead of indexing raw backend tensors
# with positions from this tuple.
IGRIS_C_JOINT_ORDER = (
    "Joint_Shoulder_Pitch_Left",
    "Joint_Shoulder_Pitch_Right",
    "Joint_Waist_Yaw",
    "Joint_Shoulder_Roll_Left",
    "Joint_Shoulder_Roll_Right",
    "Joint_Waist_Roll",
    "Joint_Shoulder_Yaw_Left",
    "Joint_Shoulder_Yaw_Right",
    "Joint_Waist_Pitch",
    "Joint_Elbow_Pitch_Left",
    "Joint_Elbow_Pitch_Right",
    "Joint_Hip_Pitch_Left",
    "Joint_Hip_Pitch_Right",
    "Joint_Hip_Roll_Left",
    "Joint_Hip_Roll_Right",
    "Joint_Hip_Yaw_Left",
    "Joint_Hip_Yaw_Right",
    "Joint_Knee_Pitch_Left",
    "Joint_Knee_Pitch_Right",
    "Joint_Ankle_Pitch_Left",
    "Joint_Ankle_Pitch_Right",
    "Joint_Ankle_Roll_Left",
    "Joint_Ankle_Roll_Right",
)
MYACTUATOR_X12_150_ARMATURE = 0.12
MYACTUATOR_X12_150_WN = 2 * math.pi * 4.0
MYACTUATOR_X12_150_DAMPING_RATIO = 2.0
MYACTUATOR_X12_150_STIFFNESS = MYACTUATOR_X12_150_ARMATURE * MYACTUATOR_X12_150_WN**2
MYACTUATOR_X12_150_DAMPING = 2 * MYACTUATOR_X12_150_ARMATURE * MYACTUATOR_X12_150_DAMPING_RATIO * MYACTUATOR_X12_150_WN
MYACTUATOR_X8_120_ARMATURE = 0.055
MYACTUATOR_X8_120_WN = 2 * math.pi * 6.0
MYACTUATOR_X8_120_DAMPING_RATIO = 2.0
MYACTUATOR_X8_120_STIFFNESS = MYACTUATOR_X8_120_ARMATURE * MYACTUATOR_X8_120_WN**2
MYACTUATOR_X8_120_DAMPING = 2 * MYACTUATOR_X8_120_ARMATURE * MYACTUATOR_X8_120_DAMPING_RATIO * MYACTUATOR_X8_120_WN
MYACTUATOR_X8_90_ARMATURE = 0.09
MYACTUATOR_X8_90_WN = 2 * math.pi * 3.0
MYACTUATOR_X8_90_DAMPING_RATIO = 2.0
MYACTUATOR_X8_90_STIFFNESS = MYACTUATOR_X8_90_ARMATURE * MYACTUATOR_X8_90_WN**2
MYACTUATOR_X8_90_DAMPING = 2 * MYACTUATOR_X8_90_ARMATURE * MYACTUATOR_X8_90_DAMPING_RATIO * MYACTUATOR_X8_90_WN
MYACTUATOR_X8_60_ARMATURE = 0.019
MYACTUATOR_X8_60_WN = 2 * math.pi * 8.0
MYACTUATOR_X8_60_DAMPING_RATIO = 2.0
MYACTUATOR_X8_60_STIFFNESS = MYACTUATOR_X8_60_ARMATURE * MYACTUATOR_X8_60_WN**2
MYACTUATOR_X8_60_DAMPING = 2 * MYACTUATOR_X8_60_ARMATURE * MYACTUATOR_X8_60_DAMPING_RATIO * MYACTUATOR_X8_60_WN

##
# Configuration - Actuators.
##

IGRISC_EXPLICIT_ACTUATOR = {
    "X12-150": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Hip_Pitch.*", ".*Knee.*"],
        # Peak speed is not listed in the datasheet; use rated speed as the hard cap for now.
        actuator_velocity_limit=10.4720,  # 12.32,
        velocity_rated=10.4720,
        actuator_effort_limit=50.0,
        torque_limit=150.0,
        stiffness={
            ".*.*": MYACTUATOR_X12_150_STIFFNESS,
        },
        damping={
            ".*.*": MYACTUATOR_X12_150_DAMPING,
        },
        armature={".*Hip_Pitch.*": 0.3, ".*Knee.*": 0.18},
        friction={".*Hip_Pitch.*": 0.5, ".*Knee.*": 0.5},
        viscous_friction={".*Hip_Pitch.*": 2.4, ".*Knee.*": 2.2},
        slack_range={
            ".*Hip_Pitch.*": (0.0, 0.02),
            ".*Knee_Pitch.*": (0.0, 0.02),
        },
        max_delay=10,  # max delay in simulation steps
    ),
    "X8-120": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Hip_Roll.*"],
        actuator_velocity_limit=13.2994,  # 16.5457,
        velocity_rated=13.2994,
        actuator_effort_limit=43.0,
        torque_limit=120.0,
        stiffness=MYACTUATOR_X8_120_STIFFNESS,
        damping=MYACTUATOR_X8_120_DAMPING,
        armature=0.06,
        friction=0.1,
        viscous_friction=3.0,
        slack_range={".*Hip_Roll.*": (0.0, 0.02)},
        max_delay=10,  # max delay in simulation steps
    ),
    "X8-60": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Hip_Yaw.*", ".*Waist_Yaw.*", ".*_Shoulder_.*", ".*_Elbow_.*"],
        actuator_velocity_limit=16.0221,  # 18.4307,
        velocity_rated=16.0221,
        actuator_effort_limit=20.0,
        torque_limit=60.0,
        stiffness={
            ".*.*": MYACTUATOR_X8_60_STIFFNESS,
        },
        damping={
            ".*.*": MYACTUATOR_X8_60_DAMPING,
        },
        armature={
            ".*.*": 0.02,
        },
        friction={
            ".*.*": 1.01,
        },
        viscous_friction={".*.*": 1.8},
        slack_range={".*.*": (0.0, 0.02)},
        max_delay=10,  # max delay in simulation steps
    ),
    "PLA": DelayedDCMotorWithSlackCfg(
        joint_names_expr=[".*Ankle.*", ".*Waist_Roll.*", ".*Waist_Pitch.*"],
        actuator_velocity_limit={
            # Max Box
            ".*Waist_Roll.*": 8.6608,  # 10.1892,
            ".*Waist_Pitch.*": 15.6655,  # 18.43,
            ".*Ankle_Pitch.*": 18.1383,  # 21.3392,
            ".*Ankle_Roll.*": 19.8934,  # 23.4040,
            # # Conservative Box
            # ".*Waist_Roll.*": 6.5597,
            # ".*Waist_Pitch.*": 6.5597,
            # ".*Ankle_Pitch.*": 11.1620,
            # ".*Ankle_Roll.*": 11.1620,
        },
        velocity_rated={
            # Max Box
            ".*Waist_Roll.*": 8.6608,
            ".*Waist_Pitch.*": 15.6655,
            ".*Ankle_Pitch.*": 18.1383,
            ".*Ankle_Roll.*": 19.8934,
            # # Conservative Box
            # ".*Waist_Roll.*": 5.5757,
            # ".*Waist_Pitch.*": 5.5757,
            # ".*Ankle_Pitch.*": 9.4877,
            # ".*Ankle_Roll.*": 9.4877,
        },
        actuator_effort_limit={
            # Max Box
            ".*Waist_Roll.*": 72.3418,
            ".*Waist_Pitch.*": 40.0,
            ".*Ankle_Pitch.*": 54.3721,
            ".*Ankle_Roll.*": 49.5750,
            # # Conservative Box
            # ".*Waist_Roll.*": 25.76,
            # ".*Waist_Pitch.*": 25.76,
            # ".*Ankle_Pitch.*": 25.93,
            # ".*Ankle_Roll.*": 25.93,
        },
        torque_limit={
            # Max Box
            ".*Waist_Roll.*": 217.0253,
            ".*Waist_Pitch.*": 120.0,
            ".*Ankle_Pitch.*": 163.1163,
            ".*Ankle_Roll.*": 148.7251,
            # # Conservative Box
            # ".*Waist_Roll.*": 77.28,
            # ".*Waist_Pitch.*": 77.28,
            # ".*Ankle_Pitch.*": 77.79,
            # ".*Ankle_Roll.*": 77.79,
        },
        stiffness={
            ".*Waist_Roll.*": 6.545 * MYACTUATOR_X8_60_STIFFNESS,
            ".*Waist_Pitch.*": 2.2843 * MYACTUATOR_X8_60_STIFFNESS,
            ".*Ankle_Pitch.*": 1.6427 * MYACTUATOR_X8_90_STIFFNESS,
            ".*Ankle_Roll.*": 1.3656 * MYACTUATOR_X8_90_STIFFNESS,
        },  # P gain in Nm/rad
        damping={
            ".*Waist_Roll.*": 6.545 * MYACTUATOR_X8_60_DAMPING,
            ".*Waist_Pitch.*": 2.2843 * MYACTUATOR_X8_60_DAMPING,
            ".*Ankle_Pitch.*": 1.6427 * MYACTUATOR_X8_90_DAMPING,
            ".*Ankle_Roll.*": 1.3656 * MYACTUATOR_X8_90_DAMPING,
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
        slack_range={
            ".*Waist_Roll.*": (0.0, 0.02),
            ".*Waist_Pitch.*": (0.0, 0.02),
            ".*Ankle_Pitch.*": (0.0, 0.02),
            ".*Ankle_Roll.*": (0.0, 0.02),
        },
        max_delay=10,  # max delay in simulation steps
    ),
}
"""Configuration for MyActuator actuators with Explicit actuator model."""

##
# Configuration - Articulation.
##

_ROBROS_ASSET_DIR = Path(__file__).resolve().parent / "robros"
_USD_SUBDIR = Path("from_urdf") / "usd"

_IGRIS_C_USD_ENTRY_POINTS = {
    "locomotion": Path("extras/locomotion/igris_c_v2/igris_c_v2.usda"),
}


def _igris_c_usd_path(variant: Literal["locomotion"]) -> str:
    """Return the explicitly configured Isaac Sim 6 USD entry point."""
    relative_path = _IGRIS_C_USD_ENTRY_POINTS[variant]
    usd_path = _ROBROS_ASSET_DIR / "igris_c" / _USD_SUBDIR / relative_path
    if not usd_path.is_file():
        raise FileNotFoundError(f"Configured robot USD entry point does not exist: {usd_path}")
    return str(usd_path)


def _igris_c_articulation_cfg(**kwargs) -> ArticulationCfg:
    """Create an IGRIS-C articulation with a backend-independent public tensor order."""
    return ArticulationCfg(
        # Preserve the established PhysX-facing policy/dataset joint contract exactly.
        joint_ordering=IGRIS_C_JOINT_ORDER,
        # IGRIS-C body-indexed interfaces historically follow the PhysX articulation view.
        body_ordering="physx",
        **kwargs,
    )


IGRIS_C_LOCOMOTION_CFG = _igris_c_articulation_cfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=_igris_c_usd_path("locomotion"),
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
            enabled_self_collisions=True, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        ),
        # collision_props=sim_utils.CollisionPropertiesCfg(contact_offset=0.02, rest_offset=0.0),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.98),
        joint_pos={
            # Legs (12)
            ".*_Hip_Pitch_.*": -0.2,
            ".*_Hip_Roll_.*": 0.0,
            ".*_Hip_Yaw_.*": 0.0,
            ".*_Knee_Pitch_.*": 0.3,
            ".*_Ankle_Pitch_.*": -0.1,
            ".*_Ankle_Roll_.*": 0.0,
            # Waist (3)
            ".*_Waist_.*": 0.0,
            # Left arm (4)
            ".*_Shoulder_Pitch_Left": 0.13,
            ".*_Shoulder_Roll_Left": 0.13,
            ".*_Shoulder_Yaw_Left": 0.0,
            ".*_Elbow_Pitch_Left": -0.3,
            # Right arm (4)
            ".*_Shoulder_Pitch_Right": 0.13,
            ".*_Shoulder_Roll_Right": -0.13,
            ".*_Shoulder_Yaw_Right": -0.0,
            ".*_Elbow_Pitch_Right": -0.3,
        },
        joint_vel={".*": 0.0},
    ),
    actuators=IGRISC_EXPLICIT_ACTUATOR,  # type: ignore
    soft_joint_pos_limit_factor=0.98,
)


##
# Configuration - Sensors.
##
