from __future__ import annotations

import isaaclab.utils.math as math_utils
import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.utils import configclass


class RobrosVelocityCommand(UniformVelocityCommand):
    """Uniform velocity command that computes heading control from a configurable anchor link."""

    cfg: RobrosVelocityCommandCfg

    def __init__(self, cfg: RobrosVelocityCommandCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)

        heading_anchor_body_ids, heading_anchor_body_names = self.robot.find_bodies(self.cfg.heading_anchor_name)
        if len(heading_anchor_body_ids) != 1:
            raise ValueError(
                "Only one heading anchor link can be specified. "
                f"Regex '{self.cfg.heading_anchor_name}' resolved to {len(heading_anchor_body_ids)} bodies: "
                f"{heading_anchor_body_names}."
            )
        self._heading_anchor_body_idx = heading_anchor_body_ids[0]

    def _update_command(self) -> None:
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_anchor_quat_w = self.robot.data.body_quat_w[env_ids, self._heading_anchor_body_idx]
            heading_w = torch.atan2(
                2.0
                * (
                    heading_anchor_quat_w[:, 0] * heading_anchor_quat_w[:, 3]
                    + heading_anchor_quat_w[:, 1] * heading_anchor_quat_w[:, 2]
                ),
                1.0 - 2.0 * (heading_anchor_quat_w[:, 2].square() + heading_anchor_quat_w[:, 3].square()),
            )
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - heading_w)
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0


@configclass
class RobrosVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = RobrosVelocityCommand
    heading_anchor_name: str = ".*Pelvis.*"


__all__ = ["RobrosVelocityCommand", "RobrosVelocityCommandCfg"]
