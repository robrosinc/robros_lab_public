# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Common functions that can be used to activate certain terminations.

The functions can be passed to the :class:`isaaclab.managers.TerminationTermCfg` object to enable
the termination introduced by the function.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from isaaclab.assets import Articulation, RigidObject
from isaaclab.managers import SceneEntityCfg
from isaaclab.sensors import ContactSensor, RayCaster

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedRLEnv

from isaaclab.managers import ManagerTermBase, TerminationTermCfg


def root_height_below_minimum(
    env: ManagerBasedRLEnv,
    minimum_height: float,
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
    sensor_cfg: SceneEntityCfg = SceneEntityCfg("height_scanner"),
) -> torch.Tensor:
    """Terminate when the asset's root height is below the minimum height.

    Note:
        This is a custom implementation that works for both flat and rough terrains.
    """
    if sensor_cfg is not None:
        sensor: RayCaster = env.scene[sensor_cfg.name]
        # Adjust the target height using the sensor data
        adjusted_minimum_height = minimum_height + torch.mean(sensor.data.ray_hits_w[..., 2], dim=1)
    else:
        # Use the provided target height directly for flat terrain
        adjusted_minimum_height = minimum_height
    # extract the used quantities (to enable type-hinting)
    asset: RigidObject = env.scene[asset_cfg.name]
    return asset.data.root_pos_w[:, 2] < adjusted_minimum_height


def _side_pattern(pattern: str, side: str) -> str:
    if "Left/Right" in pattern or "left/right" in pattern:
        return pattern.replace("{}", ".*").replace("Left/Right", side).replace("left/right", side.lower())
    return (
        pattern.replace("{}", side)
        .replace("{side}", side)
        .replace("{SIDE}", side.upper())
        .replace("{side_lower}", side.lower())
        .replace("Left/Right", side)
        .replace("left/right", side.lower())
    )


class ThighContactAccelerationTermination(ManagerTermBase):
    """Terminate when a thigh accelerates too fast while the matching foot is in contact."""

    def __init__(self, cfg: TerminationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        params = cfg.params
        asset_cfg = params.get("asset_cfg", SceneEntityCfg("robot"))
        sensor_cfg = params.get("sensor_cfg", SceneEntityCfg("contact_forces"))
        thigh_name_pattern = params["thigh_name_pattern"]
        foot_name_pattern = params["foot_name_pattern"]
        side_names = params.get("side_names", ("Left", "Right"))

        asset: Articulation = env.scene[asset_cfg.name]
        sensor: ContactSensor = env.scene.sensors[sensor_cfg.name]
        self._asset = asset
        self._sensor = sensor
        self._acceleration_threshold_sq = float(params["acceleration_threshold"]) ** 2
        self._contact_threshold_sq = float(params.get("contact_threshold", 1.0)) ** 2
        self._body_id_pairs: list[tuple[torch.Tensor, torch.Tensor]] = []

        for side in side_names:
            thigh_ids, _ = asset.find_bodies(_side_pattern(thigh_name_pattern, side))
            foot_ids, _ = sensor.find_bodies(_side_pattern(foot_name_pattern, side))
            if len(thigh_ids) == 0 or len(foot_ids) == 0:
                raise ValueError(
                    f"Could not resolve thigh/foot bodies for side '{side}' using patterns "
                    f"'{thigh_name_pattern}' and '{foot_name_pattern}'."
                )
            self._body_id_pairs.append(
                (
                    torch.tensor(thigh_ids, device=self.device, dtype=torch.long),
                    torch.tensor(foot_ids, device=self.device, dtype=torch.long),
                )
            )

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        acceleration_threshold: float,
        thigh_name_pattern: str,
        foot_name_pattern: str,
        asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
        sensor_cfg: SceneEntityCfg = SceneEntityCfg("contact_forces"),
        contact_threshold: float = 1.0,
        side_names: tuple[str, ...] = ("Left", "Right"),
    ) -> torch.Tensor:
        terminated = torch.zeros(env.num_envs, device=self.device, dtype=torch.bool)
        body_lin_acc_w = self._asset.data.body_lin_acc_w
        net_forces_w = self._sensor.data.net_forces_w

        for thigh_ids, foot_ids in self._body_id_pairs:
            high_accel = torch.any(
                torch.sum(torch.square(body_lin_acc_w[:, thigh_ids, :]), dim=-1) > self._acceleration_threshold_sq,
                dim=-1,
            )
            in_contact = torch.any(
                torch.sum(torch.square(net_forces_w[:, foot_ids, :]), dim=-1) > self._contact_threshold_sq,
                dim=-1,
            )
            terminated |= high_accel & in_contact
        return terminated
