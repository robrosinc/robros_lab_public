"""Functions to specify the symmetry in the observation and action space for IGRIS-C."""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import torch
from tensordict import TensorDict

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

# specify the functions that are available for import
__all__ = ["compute_symmetric_states"]


@torch.no_grad()
def compute_symmetric_states(
    env: ManagerBasedRLEnv,
    obs: TensorDict | None = None,
    actions: torch.Tensor | None = None,
):
    """Augments the given observations and actions by applying symmetry transformations.

    This function creates augmented versions of the provided observations and actions by applying the
    left-right symmetry used for the public IGRIS-C locomotion tasks.

    Args:
        env: The environment instance.
        obs: The original observation tensor dictionary. Defaults to None.
        actions: The original actions tensor. Defaults to None.

    Returns:
        Augmented observations and actions tensors, or None if the respective input was None.
    """

    # observations
    if obs is not None:
        batch_size = obs.batch_size[0]
        # Original batch plus left-right mirrored batch.
        obs_aug = obs.repeat(2)

        group_names = tuple(obs.keys())
        for group_name in group_names:
            obs_aug[group_name][:batch_size] = obs[group_name][:]
            obs_aug[group_name][batch_size : 2 * batch_size] = _transform_obs_left_right(
                env.unwrapped, obs[group_name], group_name
            )
    else:
        obs_aug = None

    # actions
    if actions is not None:
        batch_size = actions.shape[0]
        # Original batch plus left-right mirrored batch.
        actions_aug = torch.zeros(batch_size * 2, actions.shape[1], device=actions.device)
        # -- original
        actions_aug[:batch_size] = actions[:]
        # -- left-right
        actions_aug[batch_size : 2 * batch_size] = _transform_actions_left_right(env.unwrapped, actions)
    else:
        actions_aug = None

    return obs_aug, actions_aug


def _transform_obs_left_right(env: ManagerBasedRLEnv, obs: torch.Tensor, group_name: str) -> torch.Tensor:
    """Apply the cached left-right symmetry transform for an observation group."""
    obs_ = obs.clone()
    transform_plan = _get_igrisc_obs_left_right_plan(env, group_name)
    if len(transform_plan) == 0:
        return obs_

    batch_size = obs_.shape[0]
    height_perm = _get_igrisc_height_left_right_perm(env)
    if height_perm is not None:
        height_perm = _to_device(height_perm, obs_.device)

    for start, end, op, width in transform_plan:
        term = obs_[:, start:end]

        if op == "flip_yz":
            block = term.reshape(batch_size, -1, width)
            block[..., 1].mul_(-1.0)
            block[..., 2].mul_(-1.0)
        elif op == "flip_xz":
            block = term.reshape(batch_size, -1, width)
            block[..., 0].mul_(-1.0)
            block[..., 2].mul_(-1.0)
        elif op == "flip_y":
            term.reshape(batch_size, -1, width)[..., 1].mul_(-1.0)
        elif op == "swap_2":
            block = term.reshape(batch_size, -1, width)
            term[:] = block[..., [1, 0]].reshape_as(term)
        elif op == "joint_obs":
            term[:] = _switch_igrisc_joints_left_right(env, term.reshape(batch_size, -1, width)).reshape_as(term)
        elif op == "joint_act":
            block = term.reshape(batch_size, -1, width)
            block[:] = _switch_igrisc_actions_left_right(env, block)
        elif op == "height" and height_perm is not None:
            term[:] = term.reshape(batch_size, -1, width).index_select(-1, height_perm).reshape_as(term)

    return obs_


def _transform_actions_left_right(env: ManagerBasedRLEnv, actions: torch.Tensor) -> torch.Tensor:
    """Apply a left-right symmetry transformation to the action tensor."""
    actions = actions.clone()
    num_joints = env.scene["robot"].num_joints
    if actions.shape[1] <= num_joints:
        actions[:] = _switch_igrisc_actions_left_right(env, actions[:])
        return actions
    joint_part = _switch_igrisc_actions_left_right(env, actions[:, :num_joints])
    actions[:, :num_joints] = joint_part
    return actions


def _get_igrisc_obs_left_right_plan(env: ManagerBasedRLEnv, group_name: str) -> tuple[tuple[int, int, str, int], ...]:
    """Build and cache the observation transform plan for a specific group."""
    if not hasattr(env, "_igrisc_obs_left_right_plans"):
        env._igrisc_obs_left_right_plans = {}
    cached = env._igrisc_obs_left_right_plans.get(group_name)
    if cached is not None:
        return cached

    term_names = env.observation_manager.active_terms.get(group_name, [])
    term_dims = env.observation_manager.group_obs_term_dim.get(group_name, [])
    robot = env.scene["robot"]
    num_joints = robot.num_joints
    height_perm = _get_igrisc_height_left_right_perm(env)

    plan = []
    offset = 0
    for name, dims in zip(term_names, term_dims):
        shape = _normalize_term_dims(dims)
        term_len = math.prod(shape)
        width = _get_igrisc_obs_term_feature_width(name, shape, num_joints, height_perm)
        op = None

        if ("velocity_commands" in name or name in ("commands", "base_velocity")) and width == 3:
            op = "flip_yz"
        elif name == "base_ang_vel" and width == 3:
            op = "flip_xz"
        elif name in ("base_lin_vel", "projected_gravity") and width == 3:
            op = "flip_y"
        elif name == "phase" and width == 2:
            op = "swap_2"
        elif ("joint_pos" in name or "joint_vel" in name or "joint_torque" in name) and width == num_joints:
            op = "joint_obs"
        elif name in ("actions", "last_action", "last_actions") and width >= num_joints:
            op = "joint_act"
        elif "contact" in name and width == 2:
            op = "swap_2"
        elif name == "height_scan" and height_perm is not None:
            op = "height"

        if op is not None and term_len % width == 0:
            plan.append((offset, offset + term_len, op, width))
        offset += term_len

    cached = tuple(plan)
    env._igrisc_obs_left_right_plans[group_name] = cached
    return cached


def _normalize_term_dims(dims: int | tuple[int, ...] | list[int] | torch.Size) -> tuple[int, ...]:
    """Convert observation term dimensions to a normalized tuple."""
    if isinstance(dims, int):
        return (dims,)
    return tuple(int(dim) for dim in dims)


def _get_igrisc_obs_term_feature_width(
    term_name: str,
    shape: tuple[int, ...],
    num_joints: int,
    height_perm: torch.Tensor | None,
) -> int:
    """Return the per-sample feature width before IsaacLab history flattening."""
    if term_name in (
        "velocity_commands",
        "base_velocity",
        "commands",
        "base_ang_vel",
        "base_lin_vel",
        "projected_gravity",
    ):
        return 3
    if term_name in ("joint_pos", "joint_vel", "joint_torques", "last_action", "last_actions"):
        return num_joints
    if term_name in ("phase", "contact_state"):
        return 2
    if term_name == "height_scan" and height_perm is not None:
        return int(height_perm.numel())
    return shape[-1]


def _get_igrisc_height_left_right_perm(env: ManagerBasedRLEnv) -> torch.Tensor | None:
    """Build and cache the left-right permutation for a flattened height scan."""
    if hasattr(env, "_igrisc_height_left_right_perm"):
        return env._igrisc_height_left_right_perm

    perm = None
    if hasattr(env, "height_perm"):
        perm = torch.as_tensor(env.height_perm, device=env.device, dtype=torch.long)
    elif "height_scanner" in env.scene.sensors:
        sensor = env.scene.sensors["height_scanner"]
        pattern = sensor.cfg.pattern_cfg
        if hasattr(pattern, "size") and hasattr(pattern, "resolution"):
            nx = int(round(pattern.size[0] / pattern.resolution)) + 1
            ny = int(round(pattern.size[1] / pattern.resolution)) + 1
            perm = torch.arange(nx * ny, device=env.device, dtype=torch.long).reshape(ny, nx).flip(0).reshape(-1)

    env._igrisc_height_left_right_perm = perm
    return perm


def _to_device(index: torch.Tensor | None, device: torch.device | str) -> torch.Tensor | None:
    if index is None or index.device == torch.device(device):
        return index
    return index.to(device=device)


def _swap_left_right_indices(data: torch.Tensor, left_idx: torch.Tensor, right_idx: torch.Tensor) -> torch.Tensor:
    """Swap paired left/right indices without changing signs."""
    left_idx = _to_device(left_idx, data.device)
    right_idx = _to_device(right_idx, data.device)
    swapped = data.clone()
    if left_idx.numel() > 0:
        swapped[..., left_idx] = data[..., right_idx]
        swapped[..., right_idx] = data[..., left_idx]
    return swapped


def _switch_igrisc_joints_left_right(env: ManagerBasedRLEnv, joint_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint data tensor."""
    maps = _get_igrisc_symmetry_maps(env)
    left_idx = maps["left_idx_obs"]
    right_idx = maps["right_idx_obs"]
    flip_idx = _to_device(maps["flip_idx_obs"], joint_data.device)

    joint_data_switched = _swap_left_right_indices(joint_data, left_idx, right_idx)
    if flip_idx.numel() > 0:
        joint_data_switched[..., flip_idx] *= -1.0
    return joint_data_switched


def _switch_igrisc_actions_left_right(env: ManagerBasedRLEnv, action_data: torch.Tensor) -> torch.Tensor:
    """Applies a left-right symmetry transformation to the joint action tensor."""
    maps = _get_igrisc_symmetry_maps(env)
    left_idx = maps["left_idx_act"]
    right_idx = maps["right_idx_act"]
    flip_idx = _to_device(maps["flip_idx_act"], action_data.device)

    action_data_switched = _swap_left_right_indices(action_data, left_idx, right_idx)
    if flip_idx.numel() > 0:
        action_data_switched[..., flip_idx] *= -1.0
    return action_data_switched


def _get_igrisc_symmetry_maps(env: ManagerBasedRLEnv) -> dict:
    """Build and cache symmetry indices for the IGRIS-C joints."""
    if hasattr(env, "_igrisc_symmetry_maps"):
        return env._igrisc_symmetry_maps

    robot = env.scene["robot"]
    obs_joint_names = robot.joint_names
    action_joint_names = obs_joint_names
    action_terms = getattr(getattr(env, "action_manager", None), "_terms", None)
    if action_terms is not None:
        joint_pos_term = action_terms.get("joint_pos")
        if joint_pos_term is not None and hasattr(joint_pos_term, "_joint_names"):
            action_joint_names = joint_pos_term._joint_names

    def _build_left_right_idx(joint_names: list[str] | tuple[str, ...]):
        name_to_idx = {name: i for i, name in enumerate(joint_names)}
        left_idx = []
        right_idx = []
        for name, idx in name_to_idx.items():
            if "Left" in name:
                mirror = name.replace("Left", "Right")
                mirror_idx = name_to_idx.get(mirror)
                if mirror_idx is not None:
                    left_idx.append(idx)
                    right_idx.append(mirror_idx)
        return left_idx, right_idx

    def _has_any(n: str, keys: list[str]) -> bool:
        return any(k in n for k in keys)

    flip_keys = [
        "Waist_Yaw",
        "Waist_Roll",
        "Hip_Yaw",
        "Hip_Roll",
        "Ankle_Roll",
        "Shoulder_Roll",
        "Shoulder_Yaw",
    ]

    left_idx_obs, right_idx_obs = _build_left_right_idx(obs_joint_names)
    left_idx_act, right_idx_act = _build_left_right_idx(action_joint_names)
    flip_idx_obs = [i for i, n in enumerate(obs_joint_names) if _has_any(n, flip_keys)]
    flip_idx_act = [i for i, n in enumerate(action_joint_names) if _has_any(n, flip_keys)]

    env._igrisc_symmetry_maps = {
        "left_idx_obs": torch.tensor(left_idx_obs, device=env.device, dtype=torch.long),
        "right_idx_obs": torch.tensor(right_idx_obs, device=env.device, dtype=torch.long),
        "left_idx_act": torch.tensor(left_idx_act, device=env.device, dtype=torch.long),
        "right_idx_act": torch.tensor(right_idx_act, device=env.device, dtype=torch.long),
        "flip_idx_obs": torch.tensor(flip_idx_obs, device=env.device, dtype=torch.long),
        "flip_idx_act": torch.tensor(flip_idx_act, device=env.device, dtype=torch.long),
    }
    return env._igrisc_symmetry_maps
