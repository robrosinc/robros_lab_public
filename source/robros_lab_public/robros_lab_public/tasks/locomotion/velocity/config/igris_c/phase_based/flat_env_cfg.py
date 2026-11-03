# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from isaaclab.utils.configclass import configclass

from .rough_env_cfg import IGRISCRoughEnvCfg


@configclass
class IGRISCFlatEnvCfg(IGRISCRoughEnvCfg):
    def __post_init__(self):
        # post init of parent
        super().__post_init__()

        # change terrain to flat
        self.scene.terrain.terrain_type = "plane"
        self.scene.terrain.terrain_generator = None
        self.rewards.base_height_exp.params["sensor_cfg"] = None
        self.terminations.base_height_low.params["sensor_cfg"] = None
        # no height scan
        self.scene.height_scanner = None
        self.observations.critic.height_scan = None
        self.observations.critic.feet_height.params["sensor_cfg"] = None
        # no terrain curriculum
        self.curriculum.terrain_levels = None


@configclass
class IGRISCFlatEnvCfg_PLAY(IGRISCFlatEnvCfg):
    def __post_init__(self) -> None:
        # post init of parent
        super().__post_init__()
        self.commands.base_velocity.ranges.lin_vel_x = (0.0, 0.0)
        self.commands.base_velocity.ranges.lin_vel_y = (0.0, 0.0)
        self.commands.base_velocity.ranges.ang_vel_z = (-0.0, 0.0)
        self.commands.base_velocity.ranges.heading = (0.0, 0.0)
        self.commands.base_velocity.heading_command = False

        # make a smaller scene for play
        self.scene.num_envs = 50
        self.scene.env_spacing = 2.5
        # disable randomization for play
        self.observations.policy.enable_corruption = False
        self.events.randomize_joint_default_pos = None
        # remove random pushing
        self.events.base_external_force_torque = None
        self.events.push_robot = None
