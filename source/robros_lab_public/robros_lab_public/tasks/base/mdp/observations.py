"""Common functions that can be used to create observation terms.

The functions can be passed to the :class:`isaaclab.managers.ObservationTermCfg` object to enable
the observation introduced by the function.
"""

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster


def _is_root_link_cfg(asset_cfg: SceneEntityCfg) -> bool:
    """Return whether the config refers to the root link instead of a specific body."""
    return asset_cfg.body_names is None and asset_cfg.body_ids == slice(None)


def current_base_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Estimate base height above terrain using height scanner if available."""
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None and sensor_cfg.name in env.scene.sensors:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        ground_z = sensor.data.ray_hits_w[..., 2].mean(dim=1)
        return (asset.data.root_pos_w[:, 2] - ground_z).unsqueeze(-1)
    return asset.data.root_pos_w[:, 2].unsqueeze(-1)


def current_joint_torques(env: ManagerBasedRLEnv, asset_cfg: SceneEntityCfg = SceneEntityCfg("robot")) -> torch.Tensor:
    """Return applied joint torques."""
    asset = env.scene[asset_cfg.name]
    return asset.data.applied_torque


def feet_contact_state(
    env: ManagerBasedRLEnv,
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
    threshold: float = 1.0,
) -> torch.Tensor:
    """Binary contact state for each foot (1.0 if in contact)."""
    sensor: ContactSensor = env.scene[sensor_cfg.name]
    forces = sensor.data.net_forces_w[:, sensor_cfg.body_ids, :]
    contact = torch.norm(forces, dim=-1) > threshold
    return contact.to(dtype=torch.float)


def current_link_height(
    env: ManagerBasedRLEnv,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg | None = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Estimate selected link height above terrain using the height scanner if available."""
    asset = env.scene[asset_cfg.name]
    if sensor_cfg is not None and sensor_cfg.name in env.scene.sensors:
        sensor: RayCaster = env.scene.sensors[sensor_cfg.name]
        ground_z = sensor.data.ray_hits_w[..., 2].mean(dim=1, keepdim=True)
    else:
        ground_z = 0.0

    if _is_root_link_cfg(asset_cfg):
        return asset.data.root_pos_w[:, 2:3] - ground_z

    body_z = asset.data.body_pos_w[:, asset_cfg.body_ids, 2]
    if body_z.ndim == 1:
        body_z = body_z.unsqueeze(-1)
    return body_z - ground_z
