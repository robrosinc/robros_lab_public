# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab_tasks.manager_based.locomotion.velocity.mdp as mdp
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils import configclass

import robros_lab_public.tasks.locomotion.velocity.mdp as robros_mdp

##
# Pre-defined configs
##
from robros_lab_public.assets import IGRIS_C_LOCOMOTION_CFG
from robros_lab_public.tasks.locomotion.velocity.velocity_env_cfg import LocomotionVelocityEnvCfg
from robros_lab_public.tasks.locomotion.velocity.velocity_env_cfg import MySceneCfg as BaseSceneCfg
from robros_lab_public.tasks.locomotion.velocity.velocity_env_cfg import TerminationsCfg as BaseTerminationsCfg

SELF_COLLISION_SENSOR_SPECS = (
    ("self_collision_torso", "base_link"),
    ("self_collision_left_forearm", "Link_Elbow_Pitch_Left"),
    ("self_collision_right_forearm", "Link_Elbow_Pitch_Right"),
    ("self_collision_left_hip_bracket", "Link_Hip_Roll_Left"),
    ("self_collision_right_hip_bracket", "Link_Hip_Roll_Right"),
    ("self_collision_left_thigh", "Link_Hip_Yaw_Left"),
    ("self_collision_right_thigh", "Link_Hip_Yaw_Right"),
    ("self_collision_left_shin", "Link_Knee_Pitch_Left"),
    ("self_collision_right_shin", "Link_Knee_Pitch_Right"),
    ("self_collision_left_foot", "Link_Ankle_Roll_Left"),
    ("self_collision_right_foot", "Link_Ankle_Roll_Right"),
)
SELF_COLLISION_SENSOR_NAMES = tuple(sensor_name for sensor_name, _ in SELF_COLLISION_SENSOR_SPECS)
SELF_COLLISION_BODY_NAMES = tuple(body_name for _, body_name in SELF_COLLISION_SENSOR_SPECS)


def _self_collision_sensor_cfg(body_name: str) -> ContactSensorCfg:
    return ContactSensorCfg(
        prim_path=f"{{ENV_REGEX_NS}}/Robot/{body_name}",
        # Use exact per-body filter paths. A wildcard like "{ENV_REGEX_NS}/Robot/.*" resolves to the
        # entire articulation and breaks PhysX filtered-contact cardinality checks.
        filter_prim_paths_expr=[
            f"{{ENV_REGEX_NS}}/Robot/{other_body_name}"
            for other_body_name in SELF_COLLISION_BODY_NAMES
            if other_body_name != body_name
        ],
        history_length=3,
        debug_vis=False,
    )


@configclass
class IGRISCScene(BaseSceneCfg):
    # IsaacLab only supports filtered contact matrices for single-body sensors.
    self_collision_torso = _self_collision_sensor_cfg(SELF_COLLISION_SENSOR_SPECS[0][1])
    self_collision_left_forearm = _self_collision_sensor_cfg(SELF_COLLISION_SENSOR_SPECS[1][1])
    self_collision_right_forearm = _self_collision_sensor_cfg(SELF_COLLISION_SENSOR_SPECS[2][1])
    self_collision_left_hip_bracket = _self_collision_sensor_cfg(SELF_COLLISION_SENSOR_SPECS[3][1])
    self_collision_right_hip_bracket = _self_collision_sensor_cfg(SELF_COLLISION_SENSOR_SPECS[4][1])
    self_collision_left_thigh = _self_collision_sensor_cfg(SELF_COLLISION_SENSOR_SPECS[5][1])
    self_collision_right_thigh = _self_collision_sensor_cfg(SELF_COLLISION_SENSOR_SPECS[6][1])
    self_collision_left_shin = _self_collision_sensor_cfg(SELF_COLLISION_SENSOR_SPECS[7][1])
    self_collision_right_shin = _self_collision_sensor_cfg(SELF_COLLISION_SENSOR_SPECS[8][1])
    self_collision_left_foot = _self_collision_sensor_cfg(SELF_COLLISION_SENSOR_SPECS[9][1])
    self_collision_right_foot = _self_collision_sensor_cfg(SELF_COLLISION_SENSOR_SPECS[10][1])


@configclass
class IGRISCRewards:
    """Reward terms for the MDP."""

    # -- task
    is_alive = RewTerm(func=mdp.is_alive, weight=3.0)
    track_lin_vel_xy_exp = RewTerm(
        func=robros_mdp.track_link_lin_vel_xy_yaw_frame_exp,
        weight=2.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Waist_Pitch.*"),
        },
    )
    track_ang_vel_z_exp = RewTerm(
        func=robros_mdp.track_link_ang_vel_z_world_exp,
        weight=1.0,
        params={
            "command_name": "base_velocity",
            "std": 0.5,
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Waist_Pitch.*"),
        },
    )
    standstill_l2 = RewTerm(
        func=robros_mdp.stand_still_joint_deviation_l1,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
        weight=-0.4,
    )
    standstill_vel_l2 = RewTerm(
        func=robros_mdp.stand_still_joint_vel_l2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
        weight=-0.05,
    )
    standstill_base_vel_l2 = RewTerm(
        func=robros_mdp.stand_still_base_velocity_l2,
        params={
            "command_name": "base_velocity",
            "asset_cfg": SceneEntityCfg("robot"),
        },
        weight=-1.0,
    )
    # -- style
    joint_deviation_hip = RewTerm(
        func=robros_mdp.joint_deviation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Hip_Yaw.*", ".*_Hip_Roll.*"])},
    )
    joint_deviation_arms = RewTerm(
        func=robros_mdp.joint_deviation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=[".*_Shoulder_.*", ".*_Elbow_.*"])},
    )
    joint_deviation_waist = RewTerm(
        func=robros_mdp.joint_deviation_l2,
        weight=-0.5,
        params={"asset_cfg": SceneEntityCfg("robot", joint_names=".*Waist.*")},
    )
    base_height_exp = RewTerm(
        func=robros_mdp.base_height_exp,
        weight=0.4,
        params={"target_height": 0.94, "std": 0.07, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    flat_orientation_exp = RewTerm(
        func=robros_mdp.flat_link_orientation_exp,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"), "std": 0.2},
        weight=0.5,
    )
    feet_distance = RewTerm(
        func=robros_mdp.feet_distance,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Ankle_Roll_.*"),
            "anchor_asset_cfg": SceneEntityCfg("robot", body_names=".*_Waist_Pitch.*"),
            "nominal_width": 0.22,
            "std": 0.1,
        },
    )
    feet_aligned = RewTerm(
        func=robros_mdp.feet_aligned,
        weight=0.1,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*_Ankle_Roll_.*"),
            "command_name": "base_velocity",
            "std": 0.25,
        },
    )
    feet_air_time = RewTerm(
        func=robros_mdp.feet_air_time,
        weight=5.0,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
            "threshold": 0.5,
        },
    )
    penalize_double_stance = RewTerm(
        func=robros_mdp.penalize_double_stance,
        weight=-2.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=[".*Ankle_Roll.*"]),
            "command_name": "base_velocity",
            "threshold": 2.0,  # [N]
        },
    )
    stumble = RewTerm(
        func=robros_mdp.stumble,
        weight=-1.5,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
        },
    )
    feet_slide = RewTerm(
        func=mdp.feet_slide,
        weight=-0.05,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Ankle_Roll.*"),
        },
    )
    # -- regularization
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.05)
    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-1.0e-3,
    )
    dof_pos_limits = RewTerm(func=robros_mdp.joint_pos_limits_binary, weight=-10.0)
    dof_acc_l2 = RewTerm(
        func=mdp.joint_acc_l2,
        weight=-5.0e-7,
    )
    dof_vel_l2 = RewTerm(
        func=mdp.joint_vel_l2,
        weight=-1.0e-3,
    )
    dof_torques_l2 = RewTerm(
        func=mdp.joint_torques_l2,
        weight=-2.0e-5,
    )
    slow_touchdown = RewTerm(
        func=robros_mdp.slow_touchdown_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Ankle_Roll.*"),
        },
    )
    feet_contact_forces = RewTerm(
        func=robros_mdp.feet_contact_forces,
        weight=-1.0e-3,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"), "threshold": 600.0},
    )
    slow_touchdown = RewTerm(
        func=robros_mdp.slow_touchdown_penalty,
        weight=-1.0,
        params={
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
            "asset_cfg": SceneEntityCfg("robot", body_names=".*Ankle_Roll.*"),
        },
    )
    self_collisions = RewTerm(
        func=robros_mdp.self_collision_cost,
        weight=-10.0,
        params={
            "sensor_names": SELF_COLLISION_SENSOR_NAMES,
            "threshold": 10.0,
        },
    )


@configclass
class IGRISCTerminations(BaseTerminationsCfg):
    """Termination terms for the MDP."""

    base_height_low = DoneTerm(
        func=robros_mdp.root_height_below_minimum,
        params={
            "minimum_height": 0.5,
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("height_scanner"),
        },
    )
    orientation = DoneTerm(func=mdp.bad_orientation, params={"limit_angle": 1.2, "asset_cfg": SceneEntityCfg("robot")})
    thigh_contact_acceleration = DoneTerm(
        func=robros_mdp.ThighContactAccelerationTermination,
        params={
            "acceleration_threshold": 150.0,
            "thigh_name_pattern": ".*Hip_Yaw_{side}.*",
            "foot_name_pattern": ".*Ankle_Roll_{side}.*",
            "asset_cfg": SceneEntityCfg("robot"),
            "sensor_cfg": SceneEntityCfg("contact_forces"),
        },
    )


@configclass
class IGRISCRoughEnvCfg(LocomotionVelocityEnvCfg):
    scene: IGRISCScene = IGRISCScene(num_envs=4096, env_spacing=2.5, replicate_physics=True)
    rewards: IGRISCRewards = IGRISCRewards()
    terminations: IGRISCTerminations = IGRISCTerminations()

    def __post_init__(self):
        # post init of parent
        super().__post_init__()
        # Scene
        self.scene.robot = IGRIS_C_LOCOMOTION_CFG.replace(prim_path="{ENV_REGEX_NS}/Robot")
        if self.scene.height_scanner:
            self.scene.height_scanner.prim_path = "{ENV_REGEX_NS}/Robot/base_link"

        # Commands
        self.commands.base_velocity.ranges.lin_vel_x = (-0.5, 1.0)
        self.commands.base_velocity.ranges.lin_vel_y = (-0.5, 0.5)
        self.commands.base_velocity.ranges.ang_vel_z = (-1.0, 1.0)
        self.commands.base_velocity.heading_anchor_name = ".*Waist_Pitch.*"


@configclass
class IGRISCRoughEnvCfg_PLAY(IGRISCRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # make a smaller scene for play
        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        # spawn the robot randomly in the grid (instead of their terrain levels)
        self.scene.terrain.max_init_terrain_level = None
        # reduce the number of terrains to save memory
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
