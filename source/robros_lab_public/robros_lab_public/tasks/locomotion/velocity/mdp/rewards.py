from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import quat_apply_inverse, wrap_to_pi, yaw_quat

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def track_link_lin_vel_xy_yaw_frame_exp(
    env, std: float, command_name: str, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking commanded planar velocity in the selected link yaw frame."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    link_id = asset_cfg.body_ids
    if link_id is None:
        quat_w = asset.data.root_quat_w
        lin_vel_w = asset.data.root_lin_vel_w[:, :3]
    else:
        if len(link_id) != 1:
            raise ValueError(f"Link id is ambiguous, {link_id}")
        quat_w = asset.data.body_quat_w[:, link_id, :].squeeze(1)
        lin_vel_w = asset.data.body_lin_vel_w[:, link_id, :3].squeeze(1)

    vel_yaw = quat_apply_inverse(yaw_quat(quat_w), lin_vel_w)
    lin_vel_error = torch.sum(torch.square(command[:, :2] - vel_yaw[:, :2]), dim=1)
    return torch.exp(-lin_vel_error / std**2)


def track_link_ang_vel_z_world_exp(
    env, command_name: str, std: float, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")
) -> torch.Tensor:
    """Reward tracking commanded yaw rate with a selected link angular velocity."""
    asset = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    link_id = asset_cfg.body_ids
    if link_id is None:
        ang_vel_w = asset.data.root_ang_vel_w[:, :3]
    else:
        if len(link_id) != 1:
            raise ValueError(f"Link id is ambiguous, {link_id}")
        ang_vel_w = asset.data.body_ang_vel_w[:, link_id, 2].squeeze(1)

    ang_vel_error = torch.square(command[:, 2] - ang_vel_w)
    return torch.exp(-ang_vel_error / std**2)


def get_phase(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Read the cached global gait phase from the policy observation term."""
    policy_term_names = env.observation_manager.active_terms["policy"]
    policy_term_cfgs = env.observation_manager._group_obs_term_cfgs["policy"]

    for term_name, term_cfg in zip(policy_term_names, policy_term_cfgs, strict=True):
        if term_name != "phase":
            continue
        if not hasattr(term_cfg.func, "global_gait_phase"):
            raise AttributeError("The 'phase' observation term does not expose a cached 'global_gait_phase' state.")
        global_gait_phase = term_cfg.func.global_gait_phase  # type: ignore[attr-defined]
        break
    else:
        raise KeyError("The policy observation group does not define a 'phase' observation term.")

    velcmd = env.command_manager.get_command("base_velocity")
    slow = torch.norm(velcmd[:, :3], dim=-1) < 0.2
    phase = torch.cat((global_gait_phase.unsqueeze(dim=-1), global_gait_phase.unsqueeze(dim=-1) + 0.5), dim=-1)
    phase[slow] = 0.25
    return phase


def contact_swing_track(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*Ankle_Roll.*"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    command_name: str = "base_velocity",
    sigma: float = 0.04,
) -> torch.Tensor:
    """Phase-based contact/velocity tracking reward."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]

    phase = get_phase(env)
    norm_dist = torch.distributions.Normal(loc=0.0, scale=1.0)
    contact_prob = norm_dist.cdf(phase / sigma) * (1.0 - norm_dist.cdf((phase - 0.5) / sigma)) + norm_dist.cdf(
        (phase - 1.0) / sigma
    ) * (1.0 - norm_dist.cdf((phase - 1.5) / sigma))
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    vel = asset.data.body_lin_vel_w[:, sensor_cfg.body_ids, :]
    term_contact = 1.0 - torch.exp(-forces.square().sum(dim=-1) / 50.0)
    term_swing = 1.0 - torch.exp(-vel.square().sum(dim=-1) / 5.0)
    moving_command = torch.norm(env.command_manager.get_command(command_name)[:, :3], dim=-1) > 0.1
    return torch.sum(-(1.0 - contact_prob) * term_contact - contact_prob * term_swing, dim=-1) * moving_command


def gait_action_limit(
    env: ManagerBasedRLEnv,
    action_term: str = "gait_action",
    min_action: float = 0.7,
    max_action: float = 1.3,
) -> torch.Tensor:
    """Penalize gait actions outside a configurable closed interval."""
    raw_f = env.action_manager.get_term(action_term).processed_actions.squeeze(-1)
    return ((raw_f < min_action) | (raw_f > max_action)).to(dtype=torch.float)


def arm_swing(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    arms_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_Shoulder_.*", ".*_Elbow_.*"]),
    swing_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", joint_names=[".*_Shoulder_Pitch_.*"]),
    command_name: str = "base_velocity",
) -> torch.Tensor:
    """Reward coordinated arm swing using the shared gait phase."""
    asset = env.scene[asset_cfg.name]
    target_angle = asset.data.default_joint_pos[:, asset_cfg.joint_ids].clone().expand(env.num_envs, -1)

    phase = get_phase(env)
    if len(swing_asset_cfg.joint_ids) != 2:
        raise ValueError("Current arm_swing reward supports exactly one swing joint per arm.")

    target_angle[:, swing_asset_cfg.joint_ids] += (
        0.1 * torch.cos(2 * torch.pi * phase) * torch.sign(env.command_manager.get_command(command_name)[:, 0])[:, None]
    )
    error = asset.data.joint_pos[:, arms_asset_cfg.joint_ids] - target_angle[:, arms_asset_cfg.joint_ids]
    return torch.exp(-torch.sum(error.square(), dim=-1) / 0.1)


def feet_distance(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*Foot.*"]),
    anchor_asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=".*Pelvis.*"),
    nominal_width: float = 0.2,
    std: float = 0.1,
) -> torch.Tensor:
    """Reward nominal lateral foot spacing in the anchor yaw frame."""
    asset = env.scene[asset_cfg.name]
    feet_pos = asset.data.body_link_pos_w[:, asset_cfg.body_ids]
    distance_vector_in_global = feet_pos[:, 0] - feet_pos[:, 1]
    if anchor_asset_cfg.body_ids is None:
        anchor_yaw_quat = yaw_quat(asset.data.root_quat_w)
    else:
        if len(anchor_asset_cfg.body_ids) != 1:
            raise ValueError(f"Anchor link is ambiguous: {anchor_asset_cfg.body_ids}")
        anchor_yaw_quat = yaw_quat(asset.data.body_quat_w[:, anchor_asset_cfg.body_ids, :]).squeeze(1)

    distance_vector_in_anchor = quat_apply_inverse(anchor_yaw_quat, distance_vector_in_global)
    too_close = torch.clamp(nominal_width - distance_vector_in_anchor[:, 1].abs(), min=0.0)
    return torch.exp(-(too_close**2) / std**2)


def feet_aligned(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot", body_names=[".*Foot.*"]),
    command_name: str = "base_velocity",
    std: float = 0.25,
) -> torch.Tensor:
    """Reward feet yaw alignment when commanded yaw velocity is small."""
    asset = env.scene[asset_cfg.name]
    feet_quat = yaw_quat(asset.data.body_quat_w[:, asset_cfg.body_ids])
    w, x, y, z = feet_quat[..., 0], feet_quat[..., 1], feet_quat[..., 2], feet_quat[..., 3]
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = torch.atan2(siny_cosp, cosy_cosp)
    yaw_diff = wrap_to_pi(yaw[:, 0] - yaw[:, 1])
    small_yaw_cmd = torch.abs(env.command_manager.get_command(command_name)[:, 2]) < 0.2
    return torch.exp(-(yaw_diff**2) / std**2) * small_yaw_cmd


def penalize_double_stance(
    env: ManagerBasedRLEnv,
    command_name: str = "base_velocity",
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces", body_names=".*Foot.*"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Penalize both tracked feet being in contact while locomotion is commanded."""
    asset = env.scene["robot"]
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    command = env.command_manager.get_command(command_name)

    contact = torch.norm(sensor.data.net_forces_w[:, sensor_cfg.body_ids, :], dim=-1) > threshold
    double_stance = contact.all(dim=-1)
    moving_command = torch.norm(command[:, :3], dim=-1) > 0.1
    not_moving = torch.norm(asset.data.root_lin_vel_w[:, :3], dim=-1) < 0.2
    return (double_stance & moving_command & not_moving).float()


def stand_still_joint_deviation_l1(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize offsets from the default joint positions when the command is very small."""
    command = env.command_manager.get_command(command_name)
    command_still = torch.norm(command[:, :3], dim=1) < command_threshold
    # Penalize motion when command is nearly zero.
    return mdp.joint_deviation_l1(env, asset_cfg) * command_still


def stand_still_joint_vel_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize joint velocities when the commanded velocity vector is zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_still = torch.norm(command[:, :3], dim=1) < command_threshold
    return torch.sum(torch.square(asset.data.joint_vel[:, asset_cfg.joint_ids]), dim=1) * command_still


def stand_still_base_velocity_l2(
    env: ManagerBasedRLEnv,
    command_name: str,
    command_threshold: float = 0.1,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Penalize base linear and angular velocity when the commanded velocity vector is zero."""
    asset: Articulation = env.scene[asset_cfg.name]
    command = env.command_manager.get_command(command_name)
    command_still = torch.norm(command[:, :3], dim=1) < command_threshold
    lin_vel_l2 = torch.sum(torch.square(asset.data.root_lin_vel_b[:, :3]), dim=1)
    ang_vel_l2 = torch.sum(torch.square(asset.data.root_ang_vel_b[:, :3]), dim=1)
    return (lin_vel_l2 + ang_vel_l2) * command_still


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

        forces = sensor.data.net_forces_w[:, sensor_body_ids, :]
        curr_vel = asset.data.body_lin_vel_w[:, asset_body_ids, :]

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


__all__ = [
    "arm_swing",
    "contact_swing_track",
    "feet_aligned",
    "feet_distance",
    "gait_action_limit",
    "get_phase",
    "penalize_double_stance",
    "stand_still_base_velocity_l2",
    "stand_still_joint_deviation_l1",
    "stand_still_joint_vel_l2",
    "track_link_ang_vel_z_world_exp",
    "track_link_lin_vel_xy_yaw_frame_exp",
    "slow_touchdown_penalty",
]
