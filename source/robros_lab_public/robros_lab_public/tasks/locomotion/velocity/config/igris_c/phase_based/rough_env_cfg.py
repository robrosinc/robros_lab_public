# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import isaaclab_tasks.core.velocity.mdp as mdp
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import RewardTermCfg as RewTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensorCfg
from isaaclab.utils.configclass import configclass

import robros_lab_public.tasks.locomotion.velocity.mdp as robros_mdp

from ..base.rough_env_cfg import IGRISCActions as IGRISCBaseActions
from ..base.rough_env_cfg import IGRISCObservations as IGRISCBaseObservations
from ..base.rough_env_cfg import IGRISCRewards as IGRISCBaseRewards
from ..base.rough_env_cfg import IGRISCRoughEnvCfg as IGRISCBaseRoughEnvCfg
from ..base.rough_env_cfg import IGRISCTerminations as IGRISCBaseTerminations

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
AMP_KEY_BODY_POS_NAMES = (
    "Link_Ankle_Roll_Left",
    "Link_Ankle_Roll_Right",
    "Link_Elbow_Pitch_Left",
    "Link_Elbow_Pitch_Right",
)
AMP_KEY_BODY_ROT_NAMES = (
    "Link_Elbow_Pitch_Left",
    "Link_Elbow_Pitch_Right",
)


@configclass
class IGRISCRewards:
    """Phase-based reward deltas layered on top of the base locomotion rewards."""

    """Reward terms for the MDP."""

    self_collisions = RewTerm(
        func=robros_mdp.self_collision_cost,
        weight=-10.0,
        params={
            "sensor_names": SELF_COLLISION_SENSOR_NAMES,
            "threshold": 10.0,
        },
    )
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
    base_height_exp = RewTerm(
        func=robros_mdp.base_height_exp,
        weight=0.4,
        params={"target_height": 0.94, "std": 0.07, "sensor_cfg": SceneEntityCfg("height_scanner")},
    )
    smoothness = RewTerm(
        func=robros_mdp.action_smoothness_penalty,
        weight=-0.0025,
    )
    action_rate_l2 = RewTerm(func=mdp.action_rate_l2, weight=-0.01)
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
        weight=-5.0e-6,
    )
    dof_torques_rate_l2 = RewTerm(
        func=robros_mdp.torque_rate_penalty,
        weight=-1.5e-7,
    )
    dof_power_l1 = RewTerm(func=robros_mdp.joint_power_l1, weight=-2.5e-5)
    base_lin_acc_l2 = RewTerm(
        func=robros_mdp.body_lin_acc_l2,
        weight=-2.0e-3,
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names="base_link"),
        },
    )
    flat_orientation_exp = RewTerm(
        func=robros_mdp.flat_link_orientation_exp,
        params={"asset_cfg": SceneEntityCfg("robot", body_names="base_link"), "std": 0.2},
        weight=0.5,
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
    is_alive = RewTerm(func=mdp.is_alive, weight=5.0)
    feet_distance = RewTerm(
        func=robros_mdp.feet_distance,
        weight=0.2,
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
    dof_pos_limits = RewTerm(func=robros_mdp.joint_pos_limits_binary, weight=-0.5)
    action_l2 = RewTerm(
        func=mdp.action_l2,
        weight=-5.0e-4,
    )
    contact_swing_track = RewTerm(
        func=robros_mdp.contact_swing_track,
        weight=1.0,
        params={"sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*")},
    )
    gait_action_limit = RewTerm(func=robros_mdp.gait_action_limit, weight=-0.25, params={"action_term": "gait_action"})
    feet_air_time = RewTerm(
        func=mdp.feet_air_time,
        weight=0.03,
        params={
            "command_name": "base_velocity",
            "sensor_cfg": SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
            "threshold": 0.4,
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
        weight=-0.5,
    )
    centroidal_ang_momentum = RewTerm(
        func=robros_mdp.centroidal_ang_momentum_l2,
        weight=0.5,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "command_name": "base_velocity",
            "cam_tracking_scale": 0.5,
            "tracking_sigma": 0.25,
            "z_tracking_weight": 3.0,
            "dcam_xy_weight": 1.0e-2,
            "reference_scale": 1.0,
            "command_threshold": 0.1,
            "only_when_moving": False,
            "use_yaw_frame": True,
        },
    )


@configclass
class IGRISCTerminations(IGRISCBaseTerminations):
    """Reuse the base termination setup unchanged."""


@configclass
class IGRISCActions(IGRISCBaseActions):
    """Base joint-position actions plus the phase-frequency action."""

    gait_action = robros_mdp.GaitFreqActionCfg(asset_name="robot", lpf_alpha=0.3)


@configclass
class IGRISCObservations(IGRISCBaseObservations):
    """Phase-based IGRIS-C observations built on top of the base IGRIS-C observation layout."""

    @configclass
    class PolicyCfg(IGRISCBaseObservations.PolicyCfg):
        phase = ObsTerm(
            func=robros_mdp.gait_phase_obs,
            params={"asset_cfg": SceneEntityCfg("robot"), "default_gait_freq": 0.8},
            history_length=5,
        )
        lpf_f = ObsTerm(func=robros_mdp.gait_action, history_length=1)

    @configclass
    class CriticCfg(IGRISCBaseObservations.CriticCfg):
        phase = ObsTerm(
            func=robros_mdp.gait_phase_obs,
            params={"asset_cfg": SceneEntityCfg("robot"), "default_gait_freq": 0.8},
        )
        lpf_f = ObsTerm(func=robros_mdp.gait_action)

    policy: PolicyCfg = PolicyCfg()
    critic: CriticCfg = CriticCfg()


@configclass
class IGRISCRoughEnvCfg(IGRISCBaseRoughEnvCfg):
    """Phase-based rough-terrain environment built on top of the base locomotion config."""

    observations: IGRISCObservations = IGRISCObservations()
    rewards: IGRISCRewards = IGRISCRewards()
    terminations: IGRISCTerminations = IGRISCTerminations()
    actions: IGRISCActions = IGRISCActions()

    def __post_init__(self):
        super().__post_init__()
        self.commands.base_velocity.low_xy_commands_thres = 1.0e-3
        self.commands.base_velocity.low_w_commands_thres = 1.0e-3


@configclass
class IGRISCRoughEnvCfg_PLAY(IGRISCRoughEnvCfg):
    def __post_init__(self):
        super().__post_init__()

        self.scene.num_envs = 2
        self.scene.env_spacing = 2.5
        self.episode_length_s = 40.0
        self.scene.terrain.max_init_terrain_level = 0
        if self.scene.terrain.terrain_generator is not None:
            self.scene.terrain.terrain_generator.num_rows = 5
            self.scene.terrain.terrain_generator.num_cols = 5
            self.scene.terrain.terrain_generator.curriculum = False

        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False
        # self.observations.policy.enable_corruption = False
        self.events.base_external_force_torque = None
        self.events.push_robot = None
