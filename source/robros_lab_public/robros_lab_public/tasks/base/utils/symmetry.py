"""Shared helpers for task-space symmetry transforms.

The simulator's articulation order is an implementation detail.  Observation
terms may instead expose an explicitly ordered selection, a joint subset, or a
motion-command order.  Symmetry transforms must therefore resolve names per
term rather than reuse one articulation-wide permutation.
"""

from __future__ import annotations

import math
from collections.abc import Sequence

import torch


def observation_term_feature_width(dims, term_cfg) -> int:
    """Return one semantic sample's width without executing the observation term."""
    width = math.prod(int(dim) for dim in dims)

    history_length = int(getattr(term_cfg, "history_length", 0) or 0)
    if history_length > 0:
        if width % history_length != 0:
            raise ValueError(f"Observation width {width} is not divisible by history length {history_length}.")
        width //= history_length

    frame_offsets = getattr(term_cfg, "params", {}).get("frame_offsets")
    if frame_offsets is not None:
        num_frames = len(tuple(frame_offsets))
        if num_frames > 0:
            if width % num_frames != 0:
                raise ValueError(f"Observation width {width} is not divisible by {num_frames} preview frames.")
            width //= num_frames

    return int(width)


def left_right_permutation(names: Sequence[str], device: str | torch.device) -> torch.Tensor:
    """Return a local permutation that swaps ``Left``/``Right`` name pairs."""
    name_to_index = {name: index for index, name in enumerate(names)}
    permutation = []
    for index, name in enumerate(names):
        if "Left" in name:
            mirror_name = name.replace("Left", "Right")
        elif "Right" in name:
            mirror_name = name.replace("Right", "Left")
        else:
            mirror_name = name
        permutation.append(name_to_index.get(mirror_name, index))
    return torch.tensor(permutation, device=device, dtype=torch.long)


def joint_flip_indices(names: Sequence[str], flip_keys: Sequence[str], device: str | torch.device) -> torch.Tensor:
    """Return local indices whose generalized coordinate changes sign when mirrored."""
    indices = [index for index, name in enumerate(names) if any(key in name for key in flip_keys)]
    return torch.tensor(indices, device=device, dtype=torch.long)


def observation_term_joint_names(env, term_cfg, *, prefer_command: bool = False) -> tuple[str, ...]:
    """Resolve the exact joint order emitted by an observation term."""
    params = getattr(term_cfg, "params", {})
    robot = env.scene["robot"]

    if prefer_command:
        command_name = params.get("command_name")
        command_terms = getattr(getattr(env, "command_manager", None), "_terms", {})
        command = command_terms.get(command_name)
        resolved_names = getattr(command, "resolved_joint_names", None)
        if resolved_names is not None:
            return tuple(resolved_names)

    asset_cfg = params.get("asset_cfg")
    if asset_cfg is None:
        asset_cfg = params.get("obs_func_params", {}).get("asset_cfg")
    if asset_cfg is None:
        return tuple(robot.joint_names)

    joint_ids = asset_cfg.joint_ids
    if joint_ids is None or isinstance(joint_ids, slice):
        return tuple(robot.joint_names)
    if isinstance(joint_ids, int):
        joint_ids = [joint_ids]
    elif isinstance(joint_ids, torch.Tensor):
        joint_ids = joint_ids.tolist()
    return tuple(robot.joint_names[int(joint_id)] for joint_id in joint_ids)


def observation_term_joint_mirror_data(
    env,
    term_cfg,
    flip_keys: Sequence[str],
    *,
    prefer_command: bool = False,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build a term-local joint permutation and sign-flip index set."""
    names = observation_term_joint_names(env, term_cfg, prefer_command=prefer_command)
    return left_right_permutation(names, env.device), joint_flip_indices(names, flip_keys, env.device)


def mirror_joint_data(data: torch.Tensor, permutation: torch.Tensor, flip_indices: torch.Tensor) -> torch.Tensor:
    """Mirror joint data whose last dimension follows the supplied local ordering."""
    permutation = permutation.to(device=data.device)
    flip_indices = flip_indices.to(device=data.device)
    mirrored = data.index_select(-1, permutation)
    if flip_indices.numel() > 0:
        mirrored[..., flip_indices] *= -1.0
    return mirrored


def swap_ordered_data(data: torch.Tensor, permutation: torch.Tensor) -> torch.Tensor:
    """Apply a name-derived permutation without sign changes."""
    return data.index_select(-1, permutation.to(device=data.device))
