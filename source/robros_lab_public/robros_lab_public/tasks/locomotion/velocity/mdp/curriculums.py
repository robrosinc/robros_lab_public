# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to create curriculum for the learning environment.

The functions can be passed to the :class:`isaaclab.managers.CurriculumTermCfg` object to enable
the curriculum introduced by the function.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv


def command_levels_vel(env: ManagerBasedRLEnv, env_ids: Sequence[int]) -> dict[str, float]:
    cmd_term = env.command_manager._terms["base_velocity"]
    ranges = cmd_term.cfg.ranges

    def _current_state() -> dict[str, float]:
        return {
            "lin_vel_x_min": float(ranges.lin_vel_x[0]),
            "lin_vel_x_max": float(ranges.lin_vel_x[1]),
        }

    curriculum_interval = int(getattr(cmd_term.cfg, "curriculum_interval", env.max_episode_length))
    if curriculum_interval <= 0:
        return _current_state()

    current_interval = env.common_step_counter // curriculum_interval
    last_interval = getattr(env, "_command_levels_vel_last_interval", -1)
    if current_interval == 0 or current_interval == last_interval:
        return _current_state()
    env._command_levels_vel_last_interval = current_interval

    if isinstance(env_ids, slice):
        env_ids = torch.arange(env.num_envs, device=env.device, dtype=torch.long)[env_ids]
    else:
        env_ids = torch.as_tensor(env_ids, device=env.device, dtype=torch.long)
    if env_ids.numel() == 0:
        return _current_state()

    term_name = None
    for name in env.reward_manager.active_terms:
        if "track_lin_vel" in name:
            term_name = name
            break
    if term_name is None or term_name not in env.reward_manager._episode_sums:
        return _current_state()

    reward_weight = abs(float(env.reward_manager.get_term_cfg(term_name).weight))
    ep_sums = env.reward_manager._episode_sums[term_name][env_ids]
    mean_tracking_reward = torch.mean(ep_sums) / env.max_episode_length_s

    if mean_tracking_reward > 0.8 * reward_weight:
        max_curr = getattr(cmd_term.cfg, "max_curriculum", None)
        if max_curr is None:
            max_curr = getattr(env.cfg.commands, "max_curriculum", None)
        if max_curr is None:
            return _current_state()
        max_curr = float(max_curr)
        new_min = float(torch.clamp(torch.tensor(ranges.lin_vel_x[0] - 0.2), min=-max_curr, max=0.0))
        new_max = float(torch.clamp(torch.tensor(ranges.lin_vel_x[1] + 0.2), min=0.0, max=max_curr))
        ranges.lin_vel_x = (new_min, new_max)

    return _current_state()
