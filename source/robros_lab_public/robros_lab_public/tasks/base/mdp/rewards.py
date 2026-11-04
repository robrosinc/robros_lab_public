# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to define rewards for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.RewardTermCfg` object to
specify the reward function and its parameters.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def _get_actuator_effort_limits(asset: Articulation, joint_ids: slice | torch.Tensor | None = None) -> torch.Tensor:
    """Return configured actuator effort limits, falling back to sim limits for uncovered joints."""
    effort_limits = asset.data.joint_effort_limits.torch.clone()
    for actuator in asset.actuators.values():
        effort_limits[:, actuator.joint_indices] = actuator.effort_limit
    if joint_ids is None:
        return effort_limits
    return effort_limits[:, joint_ids]


def base_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w.torch[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.exp(-torch.square(asset.data.root_pos_w.torch[:, 2] - adjusted_target_height) / std**2)


def link_height_exp(
    env: ManagerBasedRLEnv,
    target_height: float,
    std: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*base_link.*"),
    sensor_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Penalize asset height from its target using L2 squared kernel.

    Note:
        For flat terrain, target height is in the world frame. For rough terrain,
        sensor readings can adjust the target height to account for the terrain.
    """
    # extract the used quantities (to enable type-hinting)
    if len(asset_cfg.body_ids) != 1:
        raise ValueError(f"link is ambiguous: {asset_cfg.body_ids}, current version only supports a single link.")

    asset: RigidObject = env.scene[asset_cfg.name]
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_target_height = target_height + torch.mean(sensor.data.ray_hits_w.torch[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_target_height = target_height
    # Compute the L2 squared penalty
    return torch.exp(
        -torch.square(asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2].squeeze() - adjusted_target_height) / std**2
    )


def flat_link_orientation_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*Pelvis.*")
) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if asset_cfg.body_ids is None:
        proj_grav = asset.data.projected_gravity_b.torch
    else:
        if len(asset_cfg.body_ids) != 1:
            raise ValueError(f"Anchor link is ambiguous: {asset_cfg.body_ids}")
        body_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids[0], :]
        gravity_dir = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
        proj_grav = quat_apply_inverse(body_quat, gravity_dir)
    return torch.sum(torch.square(proj_grav[:, :2]), dim=1)


def flat_link_orientation_exp(
    env: ManagerBasedRLEnv, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*Pelvis.*")
) -> torch.Tensor:
    """Penalize non-flat link orientation using L2 squared kernel.

    This is computed by penalizing the xy-components of the projected gravity vector.
    """
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]

    if asset_cfg.body_ids is None:
        proj_grav = asset.data.projected_gravity_b.torch
    else:
        if len(asset_cfg.body_ids) != 1:
            raise ValueError(f"Anchor link is ambiguous: {asset_cfg.body_ids}")
        body_quat = asset.data.body_quat_w.torch[:, asset_cfg.body_ids[0], :]
        gravity_dir = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
        proj_grav = quat_apply_inverse(body_quat, gravity_dir)
    return torch.exp(-torch.sum(torch.square(proj_grav[:, :2]), dim=1) / std**2)


def link_ang_vel_xy_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*Pelvis.*")
) -> torch.Tensor:
    """Penalize xy-axis base angular velocity using L2 squared kernel."""
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    if len(asset_cfg.body_ids) != 1:
        raise ValueError(f"Anchor link is ambiguous: {asset_cfg.body_ids}")
    return torch.sum(torch.square(asset.data.body_ang_vel_w.torch[:, asset_cfg.body_ids, :2]), dim=-1).squeeze()


def body_lin_acc_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize the linear acceleration of bodies using L2-kernel."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(torch.square(asset.data.body_lin_acc_w.torch[:, asset_cfg.body_ids, :]), dim=(-1, -2))


def joint_deviation_l2(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint positions that deviate from the default one."""
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    angle = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]
    )
    return torch.sum(torch.square(angle), dim=1)


def joint_power_l1(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize instantaneous absolute mechanical power across the selected joints."""
    asset: Articulation = env.scene[asset_cfg.name]
    return torch.sum(
        torch.abs(asset.data.joint_vel.torch[:, asset_cfg.joint_ids])
        * torch.abs(asset.data.applied_torque.torch[:, asset_cfg.joint_ids]),
        dim=1,
    )


def contact_power(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize power at contacts (force * velocity)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
    forces = sensor.data.net_forces_w.torch[:, body_ids, :]
    vel = asset.data.body_lin_vel_w.torch[:, body_ids, :]
    return torch.sum(torch.sum(forces * vel, dim=-1).abs(), dim=-1)


def self_collision_cost(env: ManagerBasedRLEnv, sensor_names: Sequence[str], threshold: float) -> torch.Tensor:
    """Penalize robot-vs-robot self-collisions from filtered single-body contact sensors."""
    if len(sensor_names) == 0:
        return torch.zeros(env.num_envs, device=env.device, dtype=torch.float)

    sensor_hits: list[torch.Tensor] = []
    for sensor_name in sensor_names:
        contact_sensor: ContactSensor = env.scene.sensors[sensor_name]
        force_matrix_data = contact_sensor.data.force_matrix_w_history
        # Preserve a current-frame fallback for sensors configured without history.
        if force_matrix_data is None:
            force_matrix_data = contact_sensor.data.force_matrix_w
        if force_matrix_data is None:
            raise RuntimeError(
                f"Contact sensor '{sensor_name}' must set filter_prim_paths_expr to report robot-vs-robot contacts."
            )
        force_matrix = force_matrix_data.torch

        force_mag = torch.norm(force_matrix, dim=-1)
        hit = torch.amax(force_mag, dim=tuple(range(2, force_mag.ndim))) > threshold
        sensor_hits.append(hit)

    return torch.stack(sensor_hits, dim=0).any(dim=(0, 2)).to(dtype=torch.float)


def feet_contact_forces(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
    threshold: float = 650.0,
) -> torch.Tensor:
    """Penalize excessive vertical contact force."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    fz_sum = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, 2].sum(dim=-1)
    return torch.clamp(fz_sum - threshold, min=0.0, max=400.0)


def stumble(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
) -> torch.Tensor:
    """Penalize lateral contact forces larger than vertical component."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]
    f_xy = torch.norm(forces[..., :2], dim=-1)
    f_z = torch.abs(forces[..., 2])
    return torch.any(f_xy > 2.0 * f_z, dim=1)


def joint_pos_limits_binary(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint positions if they cross the soft limits.

    This is computed as a binary value.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    # compute out of limits constraints
    out_of_limits = (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        < asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 0]
    )
    out_of_limits += (
        asset.data.joint_pos.torch[:, asset_cfg.joint_ids]
        > asset.data.soft_joint_pos_limits.torch[:, asset_cfg.joint_ids, 1]
    )
    return torch.any(out_of_limits, dim=1)


def joint_torques_normalized_l2(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize normalized joint torques applied on the articulation using L2 squared kernel.
       Normalized joint torques mean the applied torque divided by the max torque of that joint.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques
        contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    effort_limits = _get_actuator_effort_limits(asset, joint_ids)
    normalized_torque = asset.data.applied_torque.torch[:, joint_ids] / torch.clamp_min(effort_limits, 1.0e-6)
    return torch.sum(torch.square(normalized_torque), dim=1)


def joint_torques_rated(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Penalize joint torques that are over the rated torque.
    The rated torque is approximated as one-third of the torque limit.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques
        contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    rated_torque = _get_actuator_effort_limits(asset, joint_ids) / 3.0
    return torch.sum(asset.data.applied_torque.torch[:, joint_ids].abs() - rated_torque, dim=1)


def joint_torques_rated_binary(
    env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Penalize joint torques that are over the rated torque, in binary.
    The rated torque is approximated as one-third of the torque limit.

    .. note::
        Only the joints configured in :attr:`asset_cfg.joint_ids` will have their joint torques
        contribute to the term.
    """
    # extract the used quantities (to enable type-hinting)
    asset: Articulation = env.scene[asset_cfg.name]
    joint_ids = asset_cfg.joint_ids
    rated_torque = _get_actuator_effort_limits(asset, joint_ids) / 3.0
    return (asset.data.applied_torque.torch[:, joint_ids].abs() > rated_torque).any(dim=-1).float()


def feet_air_time(
    env: ManagerBasedRLEnv, command_name: str, sensor_cfg: SceneEntityCfg, threshold: float
) -> torch.Tensor:
    """Reward long steps taken by the feet using L2-kernel.

    This function rewards the agent for taking steps that are longer than a threshold. This helps ensure
    that the robot lifts its feet off the ground and takes steps. The reward is computed as the sum of
    the time for which the feet are in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    # extract the used quantities (to enable type-hinting)
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    first_contact = contact_sensor.compute_first_contact(env.step_dt)[:, sensor_cfg.body_ids]
    last_air_time = contact_sensor.data.last_air_time.torch[:, sensor_cfg.body_ids]
    reward = torch.sum((last_air_time - threshold) * first_contact, dim=1)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward


def feet_air_time_positive_biped(env, command_name: str, threshold: float, sensor_cfg: SceneEntityCfg) -> torch.Tensor:
    """Reward long steps taken by the feet for bipeds.

    This function rewards the agent for taking steps up to a specified threshold and also keep one foot at
    a time in the air.

    If the commands are small (i.e. the agent is not supposed to take a step), then the reward is zero.
    """
    contact_sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
    # compute the reward
    air_time = contact_sensor.data.current_air_time.torch[:, sensor_cfg.body_ids]
    contact_time = contact_sensor.data.current_contact_time.torch[:, sensor_cfg.body_ids]
    in_contact = contact_time > 0.0
    in_mode_time = torch.where(in_contact, contact_time, air_time)
    single_stance = torch.sum(in_contact.int(), dim=1) == 1
    reward = torch.min(torch.where(single_stance.unsqueeze(-1), in_mode_time, 0.0), dim=1)[0]
    reward = torch.clamp(reward, max=threshold)
    # no reward for zero command
    reward *= torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=1) > 0.1
    return reward


class torque_rate_penalty(ManagerTermBase):
    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.prev_torque: torch.Tensor | None = None
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    ) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        asset: Articulation = env.scene[asset_cfg.name]
        current_torque = asset.data.applied_torque.torch[:, asset_cfg.joint_ids].clone()

        if self.prev_torque is None or self.prev_torque.shape != current_torque.shape:
            self.prev_torque = torch.zeros_like(current_torque)
            self._initialized.zero_()

        uninitialized = ~self._initialized
        self.prev_torque[uninitialized] = current_torque[uninitialized]

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_torque - self.prev_torque), dim=1)

        # Update the previous actions for the next call
        self.prev_torque.copy_(current_torque)
        self._initialized.fill_(True)

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 1
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)
        self._initialized[env_ids] = False


class action_smoothness_penalty(ManagerTermBase):
    """
    A reward term for penalizing large instantaneous changes in the network action output.
    This penalty encourages smoother actions over time.
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.prev_prev_action = None
        self.prev_action = None
        # self.__name__ = "action_smoothness_penalty"

    def __call__(self, env: ManagerBasedRLEnv) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.

        Returns:
            The penalty value based on the action smoothness.
        """
        # Get the current action from the environment's action manager
        current_action = env.action_manager.action.clone()

        # If this is the first call, initialize the previous actions
        if self.prev_action is None:
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        if self.prev_prev_action is None:
            self.prev_prev_action = self.prev_action
            self.prev_action = current_action
            return torch.zeros(current_action.shape[0], device=current_action.device)

        # Compute the smoothness penalty
        penalty = torch.sum(torch.square(current_action - 2 * self.prev_action + self.prev_prev_action), dim=1)

        # Update the previous actions for the next call
        self.prev_prev_action = self.prev_action
        self.prev_action = current_action

        # Apply a condition to ignore penalty during the first few episodes
        startup_env_mask = env.episode_length_buf < 3
        penalty[startup_env_mask] = 0

        # Return the penalty scaled by the configured weight
        return penalty


class slow_touchdown_penalty(ManagerTermBase):
    """
    Penalize fast touchdown (contact appearing with high foot speed).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.last_vel = None
        self.last_contact_force = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*Ankle_Roll.*"),
    ) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.
            sensor_cfg: The feet contact sensors.
            asset_cfg: The robot feet asset.

        Returns:
            The penalty value based on the touchdown speed.
        """
        sensor: ContactSensor = env.scene[sensor_cfg.name]
        asset: Articulation = env.scene[asset_cfg.name]
        sensor_body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
        asset_body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else sensor_body_ids

        forces = sensor.data.net_forces_w.torch[:, sensor_body_ids, :]
        curr_vel = asset.data.body_lin_vel_w.torch[:, asset_body_ids, :]

        if self.last_contact_force is None or self.last_vel is None:
            self.last_contact_force = forces.clone()
            self.last_vel = curr_vel.clone()
            return torch.zeros(curr_vel.shape[0], device=curr_vel.device)

        force_sq = torch.sum(forces * forces, dim=-1)
        last_force_sq = torch.sum(self.last_contact_force * self.last_contact_force, dim=-1)
        touchdown = (force_sq > 1.0) & (last_force_sq < 1.0)

        curr_speed_sq = torch.sum(curr_vel * curr_vel, dim=-1)
        last_speed_sq = torch.sum(self.last_vel * self.last_vel, dim=-1)
        penalty = torch.sum(touchdown * torch.sqrt(torch.maximum(curr_speed_sq, last_speed_sq)), dim=-1)

        self.last_contact_force = forces.clone()
        self.last_vel = curr_vel.clone()

        startup_env_mask = env.episode_length_buf < 1
        penalty[startup_env_mask] = 0
        return penalty


class slow_touchdown_penalty_thres(ManagerTermBase):
    """
    Penalize fast touchdown (contact appearing with high foot speed).
    """

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        """Initialize the term.

        Args:
            cfg: The configuration of the reward term.
            env: The RL environment instance.
        """
        super().__init__(cfg, env)
        self.last_vel = None
        self.last_contact_force = None

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*Ankle_Roll.*"),
        threshold: float = 3.0,
    ) -> torch.Tensor:
        """Compute the action smoothness penalty.

        Args:
            env: The RL environment instance.
            sensor_cfg: The feet contact sensors.
            asset_cfg: The robot feet asset.

        Returns:
            The penalty value based on the touchdown speed.
        """
        sensor: ContactSensor = env.scene[sensor_cfg.name]
        asset: Articulation = env.scene[asset_cfg.name]
        sensor_body_ids = sensor_cfg.body_ids if sensor_cfg.body_ids is not None else slice(None)
        asset_body_ids = asset_cfg.body_ids if asset_cfg.body_ids is not None else sensor_body_ids

        forces = sensor.data.net_forces_w.torch[:, sensor_body_ids, :]
        curr_vel = asset.data.body_lin_vel_w.torch[:, asset_body_ids, :]

        if self.last_contact_force is None or self.last_vel is None:
            self.last_contact_force = forces.clone()
            self.last_vel = curr_vel.clone()
            return torch.zeros(curr_vel.shape[0], device=curr_vel.device)

        force_sq = torch.sum(forces * forces, dim=-1)
        last_force_sq = torch.sum(self.last_contact_force * self.last_contact_force, dim=-1)
        touchdown = (force_sq > 1.0) & (last_force_sq < 1.0)

        curr_speed_sq = torch.sum(curr_vel * curr_vel, dim=-1)
        last_speed_sq = torch.sum(self.last_vel * self.last_vel, dim=-1)
        penalty = torch.sum(touchdown * (torch.sqrt(torch.maximum(curr_speed_sq, last_speed_sq)) > threshold), dim=-1)

        self.last_contact_force = forces.clone()
        self.last_vel = curr_vel.clone()

        startup_env_mask = env.episode_length_buf < 1
        penalty[startup_env_mask] = 0
        return penalty
