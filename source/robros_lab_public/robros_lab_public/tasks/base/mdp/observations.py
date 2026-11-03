"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

from collections.abc import Callable, Sequence

import isaaclab.utils.noise as noise
import torch
from isaaclab.assets import Articulation
from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv
from isaaclab.envs.utils.io_descriptors import (
    generic_io_descriptor,
    record_dtype,
    record_joint_names,
    record_joint_pos_offsets,
    record_shape,
)
from isaaclab.managers import ManagerTermBase, ObservationTermCfg, SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster
from isaaclab.utils.math import quat_apply_inverse, quat_conjugate, quat_mul


def current_base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Estimate base height above terrain using height scanner if available."""
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None and sensor_cfg.name in env.scene.sensors:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        ground_z = sensor.data.ray_hits_w.torch[..., 2].mean(dim=1)
        return (asset.data.root_pos_w.torch[:, 2] - ground_z).unsqueeze(-1)
    return asset.data.root_pos_w.torch[:, 2].unsqueeze(-1)


def current_joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return applied joint torques."""
    asset = env.scene[asset_cfg.name]
    return asset.data.applied_torque.torch


@generic_io_descriptor(
    units="m/s^2", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_lin_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return base linear acceleration in the asset root frame."""
    asset = env.scene[asset_cfg.name]
    return quat_apply_inverse(asset.data.root_link_quat_w.torch, asset.data.body_lin_acc_w.torch[:, 0, :])


@generic_io_descriptor(
    units="rad/s^2", axes=["X", "Y", "Z"], observation_type="RootState", on_inspect=[record_shape, record_dtype]
)
def base_ang_acc(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return base angular acceleration in the asset root frame."""
    asset = env.scene[asset_cfg.name]
    return quat_apply_inverse(asset.data.root_link_quat_w.torch, asset.data.body_ang_acc_w.torch[:, 0, :])


def _actuator_joint_pos_feedback(asset: Articulation) -> torch.Tensor:
    """Return the joint-position feedback proxy exposed by the actuator models."""
    joint_pos_feedback = asset.data.joint_pos.torch.clone()
    for actuator in asset.actuators.values():
        actuator_feedback = getattr(actuator, "joint_pos_feedback", None)
        if actuator_feedback is None:
            continue

        initialized_mask = getattr(actuator, "_joint_pos_feedback_initialized", None)
        if initialized_mask is None:
            initialized_mask = torch.ones(asset.num_instances, dtype=torch.bool, device=asset.device)
        env_mask = initialized_mask.view(-1, 1)
        joint_indices = actuator.joint_indices

        if isinstance(joint_indices, slice):
            joint_pos_feedback = torch.where(env_mask, actuator_feedback, joint_pos_feedback)
        else:
            joint_pos_feedback[:, joint_indices] = torch.where(
                env_mask, actuator_feedback, joint_pos_feedback[:, joint_indices]
            )

    return joint_pos_feedback


def _actuator_joint_vel_feedback(asset: Articulation) -> torch.Tensor:
    """Return the joint-velocity feedback proxy exposed by the actuator models."""
    joint_vel_feedback = asset.data.joint_vel.torch.clone()
    for actuator in asset.actuators.values():
        actuator_feedback = getattr(actuator, "joint_vel_feedback", None)
        if actuator_feedback is None:
            continue

        initialized_mask = getattr(actuator, "_joint_vel_feedback_initialized", None)
        if initialized_mask is None:
            initialized_mask = torch.ones(asset.num_instances, dtype=torch.bool, device=asset.device)
        env_mask = initialized_mask.view(-1, 1)
        joint_indices = actuator.joint_indices

        if isinstance(joint_indices, slice):
            joint_vel_feedback = torch.where(env_mask, actuator_feedback, joint_vel_feedback)
        else:
            joint_vel_feedback[:, joint_indices] = torch.where(
                env_mask, actuator_feedback, joint_vel_feedback[:, joint_indices]
            )

    return joint_vel_feedback


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad"
)
def joint_pos_feedback(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return actuator-side joint feedback, falling back to simulated joint positions when unavailable."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _actuator_joint_pos_feedback(asset)[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState",
    on_inspect=[record_joint_names, record_dtype, record_shape, record_joint_pos_offsets],
    units="rad",
)
def joint_pos_feedback_rel(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return actuator-side joint feedback relative to the default joint positions."""
    asset: Articulation = env.scene[asset_cfg.name]
    feedback = _actuator_joint_pos_feedback(asset)
    return feedback[:, asset_cfg.joint_ids] - asset.data.default_joint_pos.torch[:, asset_cfg.joint_ids]


@generic_io_descriptor(
    observation_type="JointState", on_inspect=[record_joint_names, record_dtype, record_shape], units="rad/s"
)
def joint_vel_feedback(env: ManagerBasedEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return actuator-side joint velocity feedback, falling back to simulated joint velocities when unavailable."""
    asset: Articulation = env.scene[asset_cfg.name]
    return _actuator_joint_vel_feedback(asset)[:, asset_cfg.joint_ids]


def feet_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact state for each foot (1.0 if in contact)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :]
    contact = torch.norm(forces, dim=-1) > threshold
    return contact.to(dtype=torch.float)


def contact_forces_b(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return selected net contact forces expressed in the robot root frame."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    asset: Articulation = env.scene[asset_cfg.name]
    forces_w = _ensure_body_batch_dim(sensor.data.net_forces_w.torch[:, sensor_cfg.body_ids, :])
    root_quat_w = asset.data.root_link_quat_w.torch.unsqueeze(1).expand(-1, forces_w.shape[1], -1)
    return quat_apply_inverse(root_quat_w, forces_w).reshape(env.num_envs, -1)


def body_lin_vel_b(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> torch.Tensor:
    """Return selected link linear velocities expressed in the robot root frame."""
    asset: Articulation = env.scene[asset_cfg.name]
    body_lin_vel_w = _ensure_body_batch_dim(asset.data.body_link_lin_vel_w.torch[:, asset_cfg.body_ids, :])
    root_quat_w = asset.data.root_link_quat_w.torch.unsqueeze(1).expand(-1, body_lin_vel_w.shape[1], -1)
    return quat_apply_inverse(root_quat_w, body_lin_vel_w).reshape(env.num_envs, -1)


def _is_root_link_cfg(asset_cfg: SceneEntityCfg) -> bool:
    """Return whether the config refers to the root link instead of a specific body."""
    return asset_cfg.body_names is None and asset_cfg.body_ids == slice(None)


def _get_single_body_id(asset_cfg: SceneEntityCfg, obs_name: str) -> int:
    """Return exactly one resolved body id or raise if the selection is ambiguous."""
    body_ids = asset_cfg.body_ids
    if isinstance(body_ids, int):
        return body_ids
    if isinstance(body_ids, slice):
        raise ValueError(f"{obs_name} requires exactly one link, but got all bodies for '{asset_cfg.name}'.")
    if len(body_ids) != 1:
        raise ValueError(f"{obs_name} requires exactly one link, but got body ids {body_ids}.")
    return body_ids[0]


def _get_anchor_link_pose(
    asset, anchor_asset_cfg: SceneEntityCfg | None, obs_name: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return the anchor link pose in world coordinates."""
    if anchor_asset_cfg is None:
        raise ValueError(f"{obs_name} requires an explicit anchor_asset_cfg.")
    if _is_root_link_cfg(anchor_asset_cfg):
        raise ValueError(f"{obs_name} requires anchor_asset_cfg to resolve to exactly one non-root link.")

    body_id = _get_single_body_id(anchor_asset_cfg, obs_name)
    return asset.data.body_link_pos_w.torch[:, body_id, :], asset.data.body_quat_w.torch[:, body_id, :]


def _ensure_body_batch_dim(tensor: torch.Tensor) -> torch.Tensor:
    """Ensure a selected-body tensor keeps an explicit body dimension."""
    if tensor.ndim == 2:
        return tensor.unsqueeze(1)
    return tensor


def _call_wrapped_obs_without_corruption(
    env: ManagerBasedRLEnv,
    obs_func: Callable[..., torch.Tensor],
    obs_func_params: dict | None = None,
) -> torch.Tensor:
    """Evaluate a wrapped observation while temporarily disabling wrapper-owned corruption."""
    obs_func_params = {} if obs_func_params is None else obs_func_params
    if isinstance(obs_func, ManagerTermBase) and hasattr(obs_func, "set_corruption_enabled"):
        previous_enabled = getattr(obs_func, "corruption_enabled", False)
        obs_func.set_corruption_enabled(False)
        try:
            return obs_func(env, **obs_func_params).clone()
        finally:
            obs_func.set_corruption_enabled(previous_enabled)
    return obs_func(env, **obs_func_params).clone()


def _resolve_wrapper_func_noise_cfg(
    cfg: ObservationTermCfg,
) -> noise.NoiseCfg | noise.NoiseModelCfg | None:
    """Resolve wrapper-owned noise applied to the wrapped function output before the transform."""
    if "inner_noise" in cfg.params or "final_noise" in cfg.params:
        raise ValueError(
            "Wrapper observation terms now use ObservationTermCfg.noise for final output noise and "
            "params['func_noise'] for the wrapped function noise. "
            "The old 'inner_noise'/'final_noise' params are no longer supported."
        )
    return cfg.params.get("func_noise")


def _make_noise_model(
    noise_cfg: noise.NoiseCfg | noise.NoiseModelCfg | None,
    env: ManagerBasedRLEnv,
) -> noise.NoiseModel | None:
    """Instantiate a stateful noise model when needed."""
    if isinstance(noise_cfg, noise.NoiseModelCfg):
        return noise_cfg.class_type(noise_cfg, num_envs=env.num_envs, device=env.device)
    return None


def _reset_noise_model(
    noise_model: noise.NoiseModel | None,
    env_ids: Sequence[int] | None,
) -> None:
    """Reset a wrapper-owned noise model if present."""
    if noise_model is not None:
        noise_model.reset(env_ids)


def _apply_wrapper_noise(
    obs: torch.Tensor,
    noise_cfg: noise.NoiseCfg | noise.NoiseModelCfg | None,
    noise_model: noise.NoiseModel | None,
    enabled: bool,
) -> torch.Tensor:
    """Apply a wrapper-owned noise layer when enabled."""
    if not enabled or noise_cfg is None:
        return obs
    if isinstance(noise_cfg, noise.NoiseCfg):
        return noise_cfg.func(obs, noise_cfg)
    if noise_model is not None:
        return noise_model(obs)
    return obs


class noisy_low_pass_obs(ManagerTermBase):
    """Apply function noise, low-pass filter the result, then apply final output noise."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        cutoff_frequency_hz = float(cfg.params["cutoff_frequency_hz"])
        if cutoff_frequency_hz <= 0.0:
            raise ValueError(f"cutoff_frequency_hz must be positive, but got {cutoff_frequency_hz}.")

        self._alpha = 1.0 - torch.exp(torch.tensor(-2.0 * torch.pi * cutoff_frequency_hz * env.step_dt)).item()
        self._filtered_obs: torch.Tensor | None = None
        self._latest_noisy_obs: torch.Tensor | None = None
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._func_noise_cfg = _resolve_wrapper_func_noise_cfg(cfg)
        self._obs_func_params = cfg.params.get("obs_func_params", {})
        # ObservationManager recursively resolves SceneEntityCfg values in nested
        # parameter dictionaries before constructing class-based terms. Resolving
        # them again here leaves both regex-based names and concrete IDs populated,
        # which SceneEntityCfg correctly rejects for subset selectors.
        self._final_noise_cfg = cfg.noise
        self._func_noise_model = _make_noise_model(self._func_noise_cfg, env)
        self._final_noise_model = _make_noise_model(self._final_noise_cfg, env)
        self._corruption_enabled = self._func_noise_cfg is not None or self._final_noise_cfg is not None

        # The wrapper owns corruption so the observation manager does not re-apply it after filtering.
        cfg.noise = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self._initialized[env_ids] = False
        _reset_noise_model(self._func_noise_model, env_ids)
        _reset_noise_model(self._final_noise_model, env_ids)

    @property
    def latest_noisy_obs(self) -> torch.Tensor | None:
        """Return the most recent function-noisy observation sample before filtering."""
        if self._latest_noisy_obs is None:
            return None
        return self._latest_noisy_obs.clone()

    @property
    def corruption_enabled(self) -> bool:
        """Whether wrapper-owned corruption is active."""
        return self._corruption_enabled

    def set_corruption_enabled(self, enabled: bool) -> None:
        """Enable or disable wrapper-owned corruption."""
        self._corruption_enabled = bool(enabled) and (
            self._func_noise_cfg is not None or self._final_noise_cfg is not None
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        obs_func: Callable[[ManagerBasedRLEnv], torch.Tensor],
        cutoff_frequency_hz: float,
        obs_func_params: dict | None = None,
        func_noise: noise.NoiseCfg | noise.NoiseModelCfg | None = None,
    ) -> torch.Tensor:
        del cutoff_frequency_hz, func_noise, obs_func_params

        obs = _call_wrapped_obs_without_corruption(env, obs_func, self._obs_func_params)
        obs = _apply_wrapper_noise(obs, self._func_noise_cfg, self._func_noise_model, self._corruption_enabled)
        self._latest_noisy_obs = obs.clone()

        if self._filtered_obs is None or self._filtered_obs.shape != obs.shape:
            self._filtered_obs = obs.clone()
            self._initialized.zero_()

        uninitialized_mask = ~self._initialized
        if torch.any(uninitialized_mask):
            self._filtered_obs[uninitialized_mask] = obs[uninitialized_mask]
            self._initialized[uninitialized_mask] = True

        initialized_mask = ~uninitialized_mask
        if torch.any(initialized_mask):
            self._filtered_obs[initialized_mask] = (1.0 - self._alpha) * self._filtered_obs[
                initialized_mask
            ] + self._alpha * obs[initialized_mask]

        return _apply_wrapper_noise(
            self._filtered_obs.clone(),
            self._final_noise_cfg,
            self._final_noise_model,
            self._corruption_enabled,
        )


class finite_difference_obs(ManagerTermBase):
    """Apply function noise first, then a per-environment finite difference.

    Optionally, apply a per-environment first-order low-pass filter to the differentiated output,
    and finally apply a separate output-noise layer.
    """

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        if env.step_dt <= 0.0:
            raise ValueError(f"step_dt must be positive, but got {env.step_dt}.")

        self._dt = env.step_dt
        self._enable_lpf = bool(cfg.params.get("enable_lpf", False))
        self._lpf_alpha: float | None = None
        self._prev_obs: torch.Tensor | None = None
        self._filtered_obs: torch.Tensor | None = None
        self._latest_noisy_obs: torch.Tensor | None = None
        self._initialized = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self._func_noise_cfg = _resolve_wrapper_func_noise_cfg(cfg)
        self._obs_func_params = cfg.params.get("obs_func_params", {})
        # Nested scene entities were already resolved recursively by the
        # observation manager before this term was constructed.
        self._final_noise_cfg = cfg.noise
        self._func_noise_model = _make_noise_model(self._func_noise_cfg, env)
        self._final_noise_model = _make_noise_model(self._final_noise_cfg, env)
        self._corruption_enabled = self._func_noise_cfg is not None or self._final_noise_cfg is not None

        if self._enable_lpf:
            cutoff_frequency_hz = float(cfg.params.get("cutoff_frequency_hz", 0.0))
            if cutoff_frequency_hz <= 0.0:
                raise ValueError(
                    f"cutoff_frequency_hz must be positive when LPF is enabled, but got {cutoff_frequency_hz}."
                )
            self._lpf_alpha = 1.0 - torch.exp(torch.tensor(-2.0 * torch.pi * cutoff_frequency_hz * env.step_dt)).item()

        # The wrapper owns corruption so the observation manager does not re-apply it after differencing.
        cfg.noise = None

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            env_ids = slice(None)

        self._initialized[env_ids] = False
        if self._prev_obs is not None:
            self._prev_obs[env_ids] = 0.0
        if self._filtered_obs is not None:
            self._filtered_obs[env_ids] = 0.0
        _reset_noise_model(self._func_noise_model, env_ids)
        _reset_noise_model(self._final_noise_model, env_ids)

    @property
    def latest_noisy_obs(self) -> torch.Tensor | None:
        """Return the most recent function-noisy observation sample before finite differencing."""
        if self._latest_noisy_obs is None:
            return None
        return self._latest_noisy_obs.clone()

    @property
    def corruption_enabled(self) -> bool:
        """Whether wrapper-owned corruption is active."""
        return self._corruption_enabled

    def set_corruption_enabled(self, enabled: bool) -> None:
        """Enable or disable wrapper-owned corruption."""
        self._corruption_enabled = bool(enabled) and (
            self._func_noise_cfg is not None or self._final_noise_cfg is not None
        )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        obs_func: Callable[[ManagerBasedRLEnv], torch.Tensor],
        obs_func_params: dict | None = None,
        enable_lpf: bool = False,
        cutoff_frequency_hz: float = 0.0,
        func_noise: noise.NoiseCfg | noise.NoiseModelCfg | None = None,
    ) -> torch.Tensor:
        del enable_lpf, cutoff_frequency_hz, func_noise, obs_func_params

        obs = _call_wrapped_obs_without_corruption(env, obs_func, self._obs_func_params)
        obs = _apply_wrapper_noise(obs, self._func_noise_cfg, self._func_noise_model, self._corruption_enabled)
        self._latest_noisy_obs = obs.clone()

        if self._prev_obs is None or self._prev_obs.shape != obs.shape:
            self._prev_obs = obs.clone()
            self._initialized.zero_()

        differentiated_obs = torch.zeros_like(obs)
        initialized_mask = self._initialized
        if torch.any(initialized_mask):
            differentiated_obs[initialized_mask] = (obs[initialized_mask] - self._prev_obs[initialized_mask]) / self._dt

        self._prev_obs.copy_(obs)
        if not self._enable_lpf:
            self._initialized.fill_(True)
            return _apply_wrapper_noise(
                differentiated_obs,
                self._final_noise_cfg,
                self._final_noise_model,
                self._corruption_enabled,
            )

        if self._filtered_obs is None or self._filtered_obs.shape != differentiated_obs.shape:
            self._filtered_obs = differentiated_obs.clone()
            self._initialized.zero_()

        uninitialized_mask = ~self._initialized
        if torch.any(uninitialized_mask):
            self._filtered_obs[uninitialized_mask] = differentiated_obs[uninitialized_mask]
            self._initialized[uninitialized_mask] = True

        initialized_mask = ~uninitialized_mask
        if torch.any(initialized_mask):
            self._filtered_obs[initialized_mask] = (1.0 - self._lpf_alpha) * self._filtered_obs[
                initialized_mask
            ] + self._lpf_alpha * differentiated_obs[initialized_mask]

        return _apply_wrapper_noise(
            self._filtered_obs.clone(),
            self._final_noise_cfg,
            self._final_noise_model,
            self._corruption_enabled,
        )


def link_ang_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return link angular velocity in the selected link frame."""
    asset = env.scene[asset_cfg.name]
    if _is_root_link_cfg(asset_cfg):
        return asset.data.root_link_ang_vel_b.torch

    body_id = _get_single_body_id(asset_cfg, "link_ang_vel")
    quat_w = asset.data.body_quat_w.torch[:, body_id, :]
    ang_vel_w = asset.data.body_link_ang_vel_w.torch[:, body_id, :]
    return quat_apply_inverse(quat_w, ang_vel_w)


def link_lin_vel(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return link linear velocity in the selected link frame."""
    asset = env.scene[asset_cfg.name]
    if _is_root_link_cfg(asset_cfg):
        return asset.data.root_link_lin_vel_b.torch

    body_id = _get_single_body_id(asset_cfg, "link_lin_vel")
    quat_w = asset.data.body_quat_w.torch[:, body_id, :]
    lin_vel_w = asset.data.body_link_lin_vel_w.torch[:, body_id, :]
    return quat_apply_inverse(quat_w, lin_vel_w)


def link_ang_vel_global(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return link angular velocity in world coordinates."""
    asset = env.scene[asset_cfg.name]
    if _is_root_link_cfg(asset_cfg):
        return asset.data.root_link_ang_vel_w.torch

    body_id = _get_single_body_id(asset_cfg, "link_ang_vel_global")
    return asset.data.body_link_ang_vel_w.torch[:, body_id, :]


def link_lin_vel_global(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return link linear velocity in world coordinates."""
    asset = env.scene[asset_cfg.name]
    if _is_root_link_cfg(asset_cfg):
        return asset.data.root_link_lin_vel_w.torch

    body_id = _get_single_body_id(asset_cfg, "link_lin_vel_global")
    return asset.data.body_link_lin_vel_w.torch[:, body_id, :]


def link_projected_gravity(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return gravity projected into the selected link frame."""
    asset = env.scene[asset_cfg.name]
    if _is_root_link_cfg(asset_cfg):
        return asset.data.projected_gravity_b.torch

    body_id = _get_single_body_id(asset_cfg, "link_projected_gravity")
    body_quat = asset.data.body_quat_w.torch[:, body_id, :]
    gravity_dir = torch.nn.functional.normalize(asset.data.GRAVITY_VEC_W.torch, dim=-1)
    return quat_apply_inverse(body_quat, gravity_dir)


def link_positions(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    anchor_asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Return selected link positions expressed in the anchor-link frame."""
    asset = env.scene[asset_cfg.name]
    body_pos_w = _ensure_body_batch_dim(asset.data.body_link_pos_w.torch[:, asset_cfg.body_ids, :])
    anchor_pos_w, anchor_quat_w = _get_anchor_link_pose(asset, anchor_asset_cfg, "link_positions")

    rel_pos = quat_apply_inverse(
        anchor_quat_w.unsqueeze(1).expand(-1, body_pos_w.shape[1], -1),
        body_pos_w - anchor_pos_w.unsqueeze(1),
    )
    return rel_pos.reshape(rel_pos.shape[0], -1)


def link_orientations(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    anchor_asset_cfg: SceneEntityCfg | None = None,
) -> torch.Tensor:
    """Return selected link orientations expressed in the anchor-link frame."""
    asset = env.scene[asset_cfg.name]
    body_quat_w = _ensure_body_batch_dim(asset.data.body_quat_w.torch[:, asset_cfg.body_ids, :])
    _, anchor_quat_w = _get_anchor_link_pose(asset, anchor_asset_cfg, "link_orientations")

    rel_quat = quat_mul(
        quat_conjugate(anchor_quat_w).unsqueeze(1).expand(-1, body_quat_w.shape[1], -1),
        body_quat_w,
    )
    return rel_quat.reshape(rel_quat.shape[0], -1)


def current_link_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Estimate selected link height above terrain using the height scanner if available."""
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None and sensor_cfg.name in env.scene.sensors:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        ground_z = sensor.data.ray_hits_w.torch[..., 2].mean(dim=1, keepdim=True)
    else:
        ground_z = 0.0

    if _is_root_link_cfg(asset_cfg):
        return asset.data.root_pos_w.torch[:, 2:3] - ground_z

    body_z = asset.data.body_pos_w.torch[:, asset_cfg.body_ids, 2]
    if body_z.ndim == 1:
        body_z = body_z.unsqueeze(-1)
    return body_z - ground_z
