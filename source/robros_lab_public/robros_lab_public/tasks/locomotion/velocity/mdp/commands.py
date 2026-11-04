from __future__ import annotations

from collections.abc import Sequence
from dataclasses import MISSING

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import UniformVelocityCommand, UniformVelocityCommandCfg
from isaaclab.managers import CommandTerm, CommandTermCfg, SceneEntityCfg
from isaaclab.markers import VisualizationMarkers, VisualizationMarkersCfg
from isaaclab.markers.config import BLUE_ARROW_X_MARKER_CFG, GREEN_ARROW_X_MARKER_CFG, RED_ARROW_X_MARKER_CFG
from isaaclab.utils.configclass import configclass


def _yaw_from_quat(quat_w: torch.Tensor) -> torch.Tensor:
    """Return yaw angle from quaternions stored as ``(x, y, z, w)``."""
    return torch.atan2(
        2.0 * (quat_w[..., 3] * quat_w[..., 2] + quat_w[..., 0] * quat_w[..., 1]),
        1.0 - 2.0 * (quat_w[..., 1].square() + quat_w[..., 2].square()),
    )


class RobrosVelocityCommand(UniformVelocityCommand):
    r"""Velocity command with IGRIS-style heading anchoring and optional pure-axis sampling."""

    cfg: RobrosVelocityCommandCfg

    def __init__(self, cfg: RobrosVelocityCommandCfg, env):
        super().__init__(cfg, env)

        heading_anchor_body_ids, heading_anchor_body_names = self.robot.find_bodies(self.cfg.heading_anchor_name)
        if len(heading_anchor_body_ids) != 1:
            raise ValueError(
                "Only one heading anchor link can be specified. "
                f"Regex '{self.cfg.heading_anchor_name}' resolved to {len(heading_anchor_body_ids)} bodies: "
                f"{heading_anchor_body_names}."
            )
        self._heading_anchor_body_idx = heading_anchor_body_ids[0]

        self._validate_pure_axis_ratios()

        self._curriculum_track_lin_vel_sum = torch.zeros(self.num_envs, device=self.device)
        self._curriculum_track_lin_vel_count = torch.zeros(self.num_envs, device=self.device)
        self._curriculum_track_lin_vel_term_name: str | None = None
        self._curriculum_track_lin_vel_std: float | None = None
        self._curriculum_track_lin_vel_asset: Articulation | None = None
        self._curriculum_track_lin_vel_asset_cfg: SceneEntityCfg | None = None

    def _validate_pure_axis_ratios(self):
        pure_axis_ratio_sum = 0.0
        for ratio_name in ("rel_vx_envs", "rel_vy_envs", "rel_wz_envs"):
            ratio = float(getattr(self.cfg, ratio_name))
            if not 0.0 <= ratio <= 1.0:
                raise ValueError(f"{ratio_name} must be in [0, 1], got {ratio}.")
            pure_axis_ratio_sum += ratio
        if pure_axis_ratio_sum > 1.0:
            raise ValueError(
                f"The sum of rel_vx_envs, rel_vy_envs, and rel_wz_envs must be <= 1.0, got {pure_axis_ratio_sum}."
            )

    def reset(self, env_ids: Sequence[int] | None = None) -> dict[str, float]:
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids = slice(None)
        self._curriculum_track_lin_vel_sum[env_ids] = 0.0
        self._curriculum_track_lin_vel_count[env_ids] = 0.0
        return extras

    def _update_metrics(self):
        super()._update_metrics()
        self._update_curriculum_tracking_metrics()

    def _update_curriculum_tracking_metrics(self):
        if not self._resolve_curriculum_tracking_reward():
            return

        max_forward_command = max(float(self.cfg.ranges.lin_vel_x[1]), 0.0)
        if max_forward_command <= 0.0:
            return

        tracking_threshold = float(self.cfg.curriculum_tracking_command_ratio) * max_forward_command
        tracking_env_mask = self.vel_command_b[:, 0] > tracking_threshold
        if not torch.any(tracking_env_mask):
            return

        asset_cfg = self._curriculum_track_lin_vel_asset_cfg
        asset = self._curriculum_track_lin_vel_asset
        if asset_cfg is None or asset is None or self._curriculum_track_lin_vel_std is None:
            return

        link_id = asset_cfg.body_ids
        if link_id is None:
            quat_w = asset.data.root_quat_w.torch
            lin_vel_w = asset.data.root_lin_vel_w.torch[:, :3]
        else:
            if len(link_id) != 1:
                raise ValueError(f"Link id is ambiguous, {link_id}")
            quat_w = asset.data.body_quat_w.torch[:, link_id, :].squeeze(1)
            lin_vel_w = asset.data.body_lin_vel_w.torch[:, link_id, :3].squeeze(1)

        vel_yaw = math_utils.quat_apply_inverse(math_utils.yaw_quat(quat_w), lin_vel_w)
        lin_vel_error = torch.sum(torch.square(self.vel_command_b[:, :2] - vel_yaw[:, :2]), dim=1)
        tracking_reward = torch.exp(-lin_vel_error / self._curriculum_track_lin_vel_std**2)

        tracking_reward = tracking_reward[tracking_env_mask]
        self._curriculum_track_lin_vel_sum[tracking_env_mask] += tracking_reward
        self._curriculum_track_lin_vel_count[tracking_env_mask] += 1.0

    def _resolve_curriculum_tracking_reward(self) -> bool:
        if self._curriculum_track_lin_vel_std is not None:
            return True

        reward_manager = getattr(self._env, "reward_manager", None)
        if reward_manager is None:
            return False

        term_name = self._curriculum_track_lin_vel_term_name
        if term_name is None:
            for name in reward_manager.active_terms:
                if "track_lin_vel" in name:
                    term_name = name
                    self._curriculum_track_lin_vel_term_name = name
                    break
        if term_name is None:
            return False

        term_cfg = reward_manager.get_term_cfg(term_name)
        std = float(term_cfg.params["std"])
        if std <= 0.0:
            raise ValueError(f"{term_name} must use a positive std, got {std}.")

        asset_cfg = term_cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self._curriculum_track_lin_vel_std = std
        self._curriculum_track_lin_vel_asset_cfg = asset_cfg
        self._curriculum_track_lin_vel_asset = self._env.scene[asset_cfg.name]
        return True

    def _resample_command(self, env_ids):
        super()._resample_command(env_ids)

        pure_axis_ratio_sum = self.cfg.rel_vx_envs + self.cfg.rel_vy_envs + self.cfg.rel_wz_envs
        if pure_axis_ratio_sum <= 0.0:
            return

        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        active_env_ids = env_ids[~self.is_standing_env[env_ids]]
        if active_env_ids.numel() == 0:
            return

        random_values = torch.rand(active_env_ids.numel(), device=self.device)
        rel_vx_threshold = self.cfg.rel_vx_envs
        rel_vy_threshold = rel_vx_threshold + self.cfg.rel_vy_envs
        rel_wz_threshold = rel_vy_threshold + self.cfg.rel_wz_envs

        vx_env_ids = active_env_ids[random_values < rel_vx_threshold]
        vy_env_ids = active_env_ids[(random_values >= rel_vx_threshold) & (random_values < rel_vy_threshold)]
        wz_env_ids = active_env_ids[(random_values >= rel_vy_threshold) & (random_values < rel_wz_threshold)]

        self.vel_command_b[vx_env_ids, 1:] = 0.0
        self.vel_command_b[vy_env_ids, 0] = 0.0
        self.vel_command_b[vy_env_ids, 2] = 0.0
        self.vel_command_b[wz_env_ids, :2] = 0.0

        if self.cfg.heading_command:
            pure_axis_env_ids = torch.cat((vx_env_ids, vy_env_ids, wz_env_ids), dim=0)
            if pure_axis_env_ids.numel() > 0:
                self.is_heading_env[pure_axis_env_ids] = False

    def _update_command(self):
        if self.cfg.heading_command:
            env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
            heading_anchor_quat_w = self.robot.data.body_quat_w.torch[env_ids, self._heading_anchor_body_idx]
            heading_w = _yaw_from_quat(heading_anchor_quat_w)
            heading_error = math_utils.wrap_to_pi(self.heading_target[env_ids] - heading_w)
            self.vel_command_b[env_ids, 2] = torch.clip(
                self.cfg.heading_control_stiffness * heading_error,
                min=self.cfg.ranges.ang_vel_z[0],
                max=self.cfg.ranges.ang_vel_z[1],
            )

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        high_xy_commands_envs_mask = self.vel_command_b[:, 0].abs() > self.cfg.high_xy_commands_thres
        low_xy_commands_envs_mask = self.vel_command_b[:, :2].norm(dim=-1) < self.cfg.low_xy_commands_thres
        low_w_commands_envs_mask = self.vel_command_b[:, 2].abs() < self.cfg.low_w_commands_thres
        self.vel_command_b[high_xy_commands_envs_mask, 1] = 0.0
        self.vel_command_b[low_xy_commands_envs_mask, :2] = 0.0
        self.vel_command_b[low_w_commands_envs_mask, 2] = 0.0

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        anchor_pos_w = self.robot.data.body_pos_w.torch[:, self._heading_anchor_body_idx].clone()
        anchor_quat_w = self.robot.data.body_quat_w.torch[:, self._heading_anchor_body_idx]
        anchor_lin_vel_w = self.robot.data.body_lin_vel_w.torch[:, self._heading_anchor_body_idx, :3]

        base_pos_w = anchor_pos_w
        base_pos_w[:, 2] += 0.5
        base_quat_w = math_utils.yaw_quat(anchor_quat_w)
        current_xy_velocity = math_utils.quat_apply_inverse(base_quat_w, anchor_lin_vel_w)[:, :2]

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2], base_quat_w)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(current_xy_velocity, base_quat_w)

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)

    def _resolve_xy_velocity_to_arrow(
        self, xy_velocity: torch.Tensor, base_quat_w: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert planar velocity vectors into world-frame arrow markers."""
        default_scale = self.goal_vel_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(xy_velocity.shape[0], 1)
        arrow_scale[:, 0] *= torch.linalg.norm(xy_velocity, dim=1) * 3.0

        heading_angle = torch.atan2(xy_velocity[:, 1], xy_velocity[:, 0])
        zeros = torch.zeros_like(heading_angle)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, heading_angle)

        if base_quat_w is None:
            base_quat_w = self.robot.data.root_quat_w.torch
        arrow_quat = math_utils.quat_mul(base_quat_w, arrow_quat)

        return arrow_scale, arrow_quat


@configclass
class RobrosVelocityCommandCfg(UniformVelocityCommandCfg):
    class_type: type = RobrosVelocityCommand
    high_xy_commands_thres: float = 0.8
    low_xy_commands_thres: float = 0.2
    low_w_commands_thres: float = 0.1
    rel_vx_envs: float = 0.0
    rel_vy_envs: float = 0.0
    rel_wz_envs: float = 0.0
    curriculum_tracking_command_ratio: float = 0.8
    max_curriculum: float | None = None
    heading_anchor_name: str = ".*base_link.*"


class RobrosHeadingVelocityCommand(RobrosVelocityCommand):
    r"""Velocity command that exposes ``(vx, vy, heading_error)``.

    This keeps the base velocity-command sampling behavior, but replaces the yaw-rate command in the third slot
    with the wrapped heading error from the configured heading anchor to the sampled heading target.
    """

    cfg: RobrosHeadingVelocityCommandCfg

    def __init__(self, cfg: RobrosHeadingVelocityCommandCfg, env):
        CommandTerm.__init__(self, cfg, env)

        self.robot: Articulation = env.scene[cfg.asset_name]
        heading_anchor_body_ids, heading_anchor_body_names = self.robot.find_bodies(self.cfg.heading_anchor_name)
        if len(heading_anchor_body_ids) != 1:
            raise ValueError(
                "Only one heading anchor link can be specified. "
                f"Regex '{self.cfg.heading_anchor_name}' resolved to {len(heading_anchor_body_ids)} bodies: "
                f"{heading_anchor_body_names}."
            )
        self._heading_anchor_body_idx = heading_anchor_body_ids[0]

        if self.cfg.ranges.heading is None:
            raise ValueError("RobrosHeadingVelocityCommandCfg requires a heading range.")

        self.vel_command_b = torch.zeros(self.num_envs, 3, device=self.device)
        self.heading_target = torch.zeros(self.num_envs, device=self.device)
        self.is_heading_env = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.is_standing_env = torch.zeros_like(self.is_heading_env)
        self.heading_error = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_vel_xy"] = torch.zeros(self.num_envs, device=self.device)
        self.metrics["error_heading"] = torch.zeros(self.num_envs, device=self.device)

        self._curriculum_track_lin_vel_sum = torch.zeros(self.num_envs, device=self.device)
        self._curriculum_track_lin_vel_count = torch.zeros(self.num_envs, device=self.device)
        self._curriculum_track_lin_vel_term_name: str | None = None
        self._curriculum_track_lin_vel_std: float | None = None
        self._curriculum_track_lin_vel_asset: Articulation | None = None
        self._curriculum_track_lin_vel_asset_cfg: SceneEntityCfg | None = None

    def __str__(self) -> str:
        msg = "RobrosHeadingVelocityCommand:\n"
        msg += f"\tCommand dimension: {tuple(self.command.shape[1:])}\n"
        msg += f"\tResampling time range: {self.cfg.resampling_time_range}\n"
        msg += f"\tStanding probability: {self.cfg.rel_standing_envs}"
        return msg

    def _resample_command(self, env_ids):
        if len(env_ids) == 0:
            return
        env_ids = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        r = torch.empty(env_ids.numel(), device=self.device)
        self.vel_command_b[env_ids, 0] = r.uniform_(*self.cfg.ranges.lin_vel_x)
        self.vel_command_b[env_ids, 1] = r.uniform_(*self.cfg.ranges.lin_vel_y)
        self.heading_target[env_ids] = r.uniform_(*self.cfg.ranges.heading)
        self.is_heading_env[env_ids] = True
        self.is_standing_env[env_ids] = r.uniform_(0.0, 1.0) <= self.cfg.rel_standing_envs

    def _update_command(self):
        self.heading_error[:] = 0.0
        self.vel_command_b[~self.is_heading_env, 2] = 0.0

        env_ids = self.is_heading_env.nonzero(as_tuple=False).flatten()
        heading_anchor_quat_w = self.robot.data.body_quat_w.torch[env_ids, self._heading_anchor_body_idx]
        heading_w = _yaw_from_quat(heading_anchor_quat_w)
        self.heading_error[env_ids] = math_utils.wrap_to_pi(self.heading_target[env_ids] - heading_w)
        self.vel_command_b[env_ids, 2] = self.heading_error[env_ids]

        standing_env_ids = self.is_standing_env.nonzero(as_tuple=False).flatten()
        self.vel_command_b[standing_env_ids, :] = 0.0

        high_xy_commands_envs_mask = self.vel_command_b[:, 0].abs() > self.cfg.high_xy_commands_thres
        low_xy_commands_envs_mask = self.vel_command_b[:, :2].norm(dim=-1) < self.cfg.low_xy_commands_thres
        low_heading_commands_envs_mask = self.vel_command_b[:, 2].abs() < self.cfg.low_heading_thres
        self.vel_command_b[high_xy_commands_envs_mask, 1] = 0.0
        self.vel_command_b[low_xy_commands_envs_mask, :2] = 0.0
        self.heading_error[low_heading_commands_envs_mask] = 0.0
        self.vel_command_b[low_heading_commands_envs_mask, 2] = 0.0

    def _update_metrics(self):
        max_command_time = self.cfg.resampling_time_range[1]
        max_command_step = max(max_command_time / self._env.step_dt, 1.0)
        self.metrics["error_vel_xy"] += (
            torch.norm(self.vel_command_b[:, :2] - self.robot.data.root_lin_vel_b.torch[:, :2], dim=-1)
            / max_command_step
        )
        self.metrics["error_heading"] += torch.abs(self.command[:, 2]) / max_command_step
        self._update_curriculum_tracking_metrics()

    def _set_debug_vis_impl(self, debug_vis: bool):
        if debug_vis:
            if not hasattr(self, "goal_vel_visualizer"):
                self.goal_vel_visualizer = VisualizationMarkers(self.cfg.goal_vel_visualizer_cfg)
                self.current_vel_visualizer = VisualizationMarkers(self.cfg.current_vel_visualizer_cfg)
                self.target_heading_visualizer = VisualizationMarkers(self.cfg.target_heading_visualizer_cfg)
            self.goal_vel_visualizer.set_visibility(True)
            self.current_vel_visualizer.set_visibility(True)
            self.target_heading_visualizer.set_visibility(True)
        elif hasattr(self, "goal_vel_visualizer"):
            self.goal_vel_visualizer.set_visibility(False)
            self.current_vel_visualizer.set_visibility(False)
            self.target_heading_visualizer.set_visibility(False)

    def _debug_vis_callback(self, event):
        if not self.robot.is_initialized:
            return

        anchor_pos_w = self.robot.data.body_pos_w.torch[:, self._heading_anchor_body_idx].clone()
        anchor_quat_w = self.robot.data.body_quat_w.torch[:, self._heading_anchor_body_idx]
        anchor_lin_vel_w = self.robot.data.body_lin_vel_w.torch[:, self._heading_anchor_body_idx, :3]

        base_pos_w = anchor_pos_w
        base_pos_w[:, 2] += 0.5
        base_quat_w = math_utils.yaw_quat(anchor_quat_w)
        current_xy_velocity = math_utils.quat_apply_inverse(base_quat_w, anchor_lin_vel_w)[:, :2]

        vel_des_arrow_scale, vel_des_arrow_quat = self._resolve_xy_velocity_to_arrow(self.command[:, :2], base_quat_w)
        vel_arrow_scale, vel_arrow_quat = self._resolve_xy_velocity_to_arrow(current_xy_velocity, base_quat_w)
        heading_arrow_scale, heading_arrow_quat = self._resolve_heading_to_arrow()

        heading_pos_w = base_pos_w.clone()
        heading_pos_w[:, 2] += 0.15

        self.goal_vel_visualizer.visualize(base_pos_w, vel_des_arrow_quat, vel_des_arrow_scale)
        self.current_vel_visualizer.visualize(base_pos_w, vel_arrow_quat, vel_arrow_scale)
        self.target_heading_visualizer.visualize(heading_pos_w, heading_arrow_quat, heading_arrow_scale)

    def _resolve_heading_to_arrow(self) -> tuple[torch.Tensor, torch.Tensor]:
        default_scale = self.target_heading_visualizer.cfg.markers["arrow"].scale
        arrow_scale = torch.tensor(default_scale, device=self.device).repeat(self.heading_target.shape[0], 1)
        arrow_scale[self.is_standing_env | ~self.is_heading_env] = 0.0

        zeros = torch.zeros_like(self.heading_target)
        arrow_quat = math_utils.quat_from_euler_xyz(zeros, zeros, self.heading_target)
        return arrow_scale, arrow_quat


@configclass
class RobrosHeadingVelocityCommandCfg(CommandTermCfg):
    class_type: type = RobrosHeadingVelocityCommand
    asset_name: str = MISSING
    rel_standing_envs: float = 0.0

    @configclass
    class Ranges:
        lin_vel_x: tuple[float, float] = MISSING
        lin_vel_y: tuple[float, float] = MISSING
        heading: tuple[float, float] | None = None

    ranges: Ranges = MISSING
    high_xy_commands_thres: float = 0.8
    low_xy_commands_thres: float = 0.2
    low_heading_thres: float = 0.1
    curriculum_tracking_command_ratio: float = 0.8
    max_curriculum: float | None = None
    heading_anchor_name: str = ".*base_link.*"
    goal_vel_visualizer_cfg: VisualizationMarkersCfg = GREEN_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_goal"
    )
    current_vel_visualizer_cfg: VisualizationMarkersCfg = BLUE_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/velocity_current"
    )
    target_heading_visualizer_cfg: VisualizationMarkersCfg = RED_ARROW_X_MARKER_CFG.replace(
        prim_path="/Visuals/Command/heading_target"
    )

    goal_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    current_vel_visualizer_cfg.markers["arrow"].scale = (0.5, 0.5, 0.5)
    target_heading_visualizer_cfg.markers["arrow"].scale = (0.7, 0.08, 0.08)


__all__ = [
    "RobrosHeadingVelocityCommand",
    "RobrosHeadingVelocityCommandCfg",
    "RobrosVelocityCommand",
    "RobrosVelocityCommandCfg",
]
