from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation
from isaaclab.envs import mdp
from isaaclab.managers import ManagerTermBase, RewardTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor
from isaaclab.utils.math import matrix_from_quat, quat_apply_inverse, wrap_to_pi, yaw_quat

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
        quat_w = asset.data.root_quat_w.torch
        lin_vel_w = asset.data.root_lin_vel_w.torch[:, :3]
    else:
        if len(link_id) != 1:
            raise ValueError(f"Link id is ambiguous, {link_id}")
        quat_w = asset.data.body_quat_w.torch[:, link_id, :]
        lin_vel_w = asset.data.body_lin_vel_w.torch[:, link_id, :3].squeeze(1)

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
        ang_vel_w = asset.data.root_ang_vel_w.torch[:, :3]
    else:
        if len(link_id) != 1:
            raise ValueError(f"Link id is ambiguous, {link_id}")
        ang_vel_w = asset.data.body_ang_vel_w.torch[:, link_id, 2].squeeze(1)

    ang_vel_error = torch.square(command[:, 2] - ang_vel_w)
    return torch.exp(-ang_vel_error / std**2)


def track_heading_error_exp(env, command_name: str, std: float) -> torch.Tensor:
    """Reward minimizing the heading-error command component."""
    command = env.command_manager.get_command(command_name)
    heading_error = torch.square(command[:, 2])
    return torch.exp(-heading_error / std**2)


def _resolve_body_ids(asset: Articulation, asset_cfg: SceneEntityCfg) -> slice | torch.Tensor:
    """Return resolved body ids as either a slice or a device tensor."""
    body_ids = asset_cfg.body_ids
    if body_ids is None or body_ids == slice(None):
        return slice(None)
    if isinstance(body_ids, int):
        return torch.tensor([body_ids], device=asset.device, dtype=torch.long)
    return torch.tensor(body_ids, device=asset.device, dtype=torch.long)


def _compute_centroidal_quantities(
    asset: Articulation,
    body_ids: slice | torch.Tensor,
    masses: torch.Tensor,
    inertias: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Compute total CAM and composite inertia about the articulation CoM in world frame."""
    body_com_pos_w = asset.data.body_com_pos_w.torch[:, body_ids, :]
    body_com_lin_vel_w = asset.data.body_com_lin_vel_w.torch[:, body_ids, :]
    body_com_ang_vel_w = asset.data.body_com_ang_vel_w.torch[:, body_ids, :]
    body_link_quat_w = asset.data.body_link_quat_w.torch[:, body_ids, :]

    total_mass = masses.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    system_com_w = torch.sum(masses.unsqueeze(-1) * body_com_pos_w, dim=1) / total_mass
    rel_com_pos_w = body_com_pos_w - system_com_w.unsqueeze(1)

    rot_mats = matrix_from_quat(body_link_quat_w.reshape(-1, 4)).view(*body_link_quat_w.shape[:-1], 3, 3)
    inertia_w = rot_mats @ inertias @ rot_mats.transpose(-1, -2)

    rotational_cam_w = (inertia_w @ body_com_ang_vel_w.unsqueeze(-1)).squeeze(-1)
    orbital_cam_w = torch.linalg.cross(rel_com_pos_w, masses.unsqueeze(-1) * body_com_lin_vel_w, dim=-1)
    cam_w = torch.sum(rotational_cam_w + orbital_cam_w, dim=1)

    identity = torch.eye(3, device=body_com_pos_w.device, dtype=body_com_pos_w.dtype).view(1, 1, 3, 3)
    rel_norm_sq = torch.sum(rel_com_pos_w.square(), dim=-1, keepdim=True).unsqueeze(-1)
    rel_outer = rel_com_pos_w.unsqueeze(-1) @ rel_com_pos_w.unsqueeze(-2)
    parallel_axis = masses.unsqueeze(-1).unsqueeze(-1) * (rel_norm_sq * identity - rel_outer)
    composite_inertia_w = torch.sum(inertia_w + parallel_axis, dim=1)

    return cam_w, composite_inertia_w


class centroidal_ang_momentum_l2(ManagerTermBase):
    r"""Reward vertical CAM tracking and penalize horizontal CAM buildup."""

    def __init__(self, cfg: RewardTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self._body_ids = _resolve_body_ids(self.asset, self.asset_cfg)
        self._masses: torch.Tensor | None = None
        self._inertias: torch.Tensor | None = None
        self._prev_cam: torch.Tensor | None = None
        self._sync_rigid_body_properties()

    def _sync_rigid_body_properties(self, env_ids: torch.Tensor | None = None) -> None:
        masses = self.asset.data.body_mass.torch
        inertias = self.asset.data.body_inertia.torch.view(masses.shape[0], masses.shape[1], 3, 3)

        if self._masses is None or self._inertias is None or env_ids is None:
            self._masses = masses
            self._inertias = inertias
            return

        env_ids = env_ids.to(device=self.asset.device, dtype=torch.long)
        self._masses[env_ids] = masses[env_ids]
        self._inertias[env_ids] = inertias[env_ids]

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        self._sync_rigid_body_properties(env_ids)
        if self._prev_cam is None:
            return
        masses = self._masses[:, self._body_ids]
        inertias = self._inertias[:, self._body_ids]
        cam_w, _ = _compute_centroidal_quantities(self.asset, self._body_ids, masses, inertias)
        cam = quat_apply_inverse(yaw_quat(self.asset.data.root_quat_w.torch), cam_w)
        if env_ids is None:
            self._prev_cam = cam
            return
        env_ids = env_ids.to(device=self.asset.device, dtype=torch.long)
        self._prev_cam[env_ids] = cam[env_ids]

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        command_name: str = "base_velocity",
        cam_tracking_scale: float = 0.5,
        tracking_sigma: float = 0.25,
        z_tracking_weight: float = 3.0,
        dcam_xy_weight: float = 5.0e-2,
        reference_scale: float = 1.0,
        command_threshold: float = 0.1,
        only_when_moving: bool = False,
        use_yaw_frame: bool = True,
    ) -> torch.Tensor:
        del asset_cfg

        if self._masses is None or self._inertias is None:
            self._sync_rigid_body_properties()

        masses = self._masses[:, self._body_ids]
        inertias = self._inertias[:, self._body_ids]
        cam_w, composite_inertia_w = _compute_centroidal_quantities(self.asset, self._body_ids, masses, inertias)

        cam = quat_apply_inverse(yaw_quat(self.asset.data.root_quat_w.torch), cam_w) if use_yaw_frame else cam_w

        command = env.command_manager.get_command(command_name)
        kz_ref = reference_scale * composite_inertia_w[:, 2, 2] * command[:, 2]
        kz_error = (cam[:, 2] - kz_ref) / (1.0 + kz_ref.abs())
        z_tracking_reward = torch.exp(
            -torch.square(kz_error / max(cam_tracking_scale, 1.0e-6)) / max(tracking_sigma, 1.0e-6)
        )

        if self._prev_cam is None:
            dcam_xy_penalty = torch.zeros_like(z_tracking_reward)
        else:
            dcam = (cam - self._prev_cam) / max(float(env.step_dt), 1.0e-6)
            dcam_xy_buildup = torch.sum(cam[:, :2] * dcam[:, :2], dim=-1)
            dcam_xy_penalty = -torch.clamp_min(dcam_xy_buildup, 0.0)

        self._prev_cam = cam.clone()

        z_reward = z_tracking_weight * z_tracking_reward
        xy_reward = dcam_xy_weight * dcam_xy_penalty

        if only_when_moving:
            moving_command = (torch.norm(command[:, :3], dim=-1) > command_threshold).float()
            z_reward = z_reward * moving_command
            xy_reward = xy_reward * moving_command

        cam_reward = z_reward + xy_reward
        startup_env_mask = env.episode_length_buf < 2
        cam_reward[startup_env_mask] = z_reward[startup_env_mask]
        return cam_reward


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
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]
    vel = asset.data.body_lin_vel_w.torch[:, sensor_cfg.body_ids, :]
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
    target_angle = asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids].clone().expand(env.num_envs, -1)

    phase = get_phase(env)
    if len(swing_asset_cfg.joint_ids) != 2:
        raise ValueError("Current arm_swing reward supports exactly one swing joint per arm.")

    target_angle[:, swing_asset_cfg.joint_ids] += (
        0.1 * torch.cos(2 * torch.pi * phase) * torch.sign(env.command_manager.get_command(command_name)[:, 0])[:, None]
    )
    error = asset.data.joint_pos.torch[:, arms_asset_cfg.joint_ids] - target_angle[:, arms_asset_cfg.joint_ids]
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
    feet_pos = asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids]
    distance_vector_in_global = feet_pos[:, 0] - feet_pos[:, 1]
    if anchor_asset_cfg.body_ids is None:
        anchor_yaw_quat = yaw_quat(asset.data.root_quat_w.torch)
    else:
        if len(anchor_asset_cfg.body_ids) != 1:
            raise ValueError(f"Anchor link is ambiguous: {anchor_asset_cfg.body_ids}")
        anchor_yaw_quat = yaw_quat(asset.data.body_quat_w.torch[:, anchor_asset_cfg.body_ids, :]).squeeze(1)

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
    feet_quat = yaw_quat(asset.data.body_quat_w.torch[:, asset_cfg.body_ids])
    x, y, z, w = feet_quat.unbind(dim=-1)
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

    contact = torch.norm(sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :], dim=-1) > threshold
    double_stance = contact.all(dim=-1)
    moving_command = torch.norm(command[:, :3], dim=-1) > 0.1
    not_moving = torch.norm(asset.data.root_lin_vel_w.torch[:, :3], dim=-1) < 0.2
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
    return torch.sum(torch.square(asset.data.joint_vel.torch[:, asset_cfg.joint_ids]), dim=1) * command_still


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
    lin_vel_l2 = torch.sum(torch.square(asset.data.root_lin_vel_b.torch[:, :3]), dim=1)
    ang_vel_l2 = torch.sum(torch.square(asset.data.root_ang_vel_b.torch[:, :3]), dim=1)
    return (lin_vel_l2 + ang_vel_l2) * command_still


__all__ = [
    "arm_swing",
    "centroidal_ang_momentum_l2",
    "contact_swing_track",
    "feet_aligned",
    "feet_distance",
    "gait_action_limit",
    "get_phase",
    "penalize_double_stance",
    "stand_still_base_velocity_l2",
    "stand_still_joint_deviation_l1",
    "stand_still_joint_vel_l2",
    "track_heading_error_exp",
    "track_link_ang_vel_z_world_exp",
    "track_link_lin_vel_xy_yaw_frame_exp",
]
