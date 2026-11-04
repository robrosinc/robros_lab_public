"""Event terms shared across Robros tasks."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import isaaclab.utils.math as math_utils
import torch
from isaaclab.assets import Articulation
from isaaclab.envs.mdp import events as isaaclab_events
from isaaclab.managers import EventTermCfg, ManagerTermBase, SceneEntityCfg

if TYPE_CHECKING:
    from isaaclab.envs import ManagerBasedEnv, ManagerBasedRLEnv


def push_by_setting_velocity(
    env: ManagerBasedEnv,
    env_ids: torch.Tensor | None,
    velocity_range: dict[str, tuple[float, float]],
    asset_cfg: SceneEntityCfg = SceneEntityCfg("robot"),
) -> None:
    """Push an asset by adding a random root velocity.

    This mirrors Isaac Lab's event term while normalizing ``env_ids=None`` before
    indexing the root-velocity buffer.  Indexing a torch tensor with ``None``
    inserts a dimension, which is rejected by Newton's partial-write API.
    """
    asset: Articulation = env.scene[asset_cfg.name]
    if env_ids is None:
        env_ids = torch.arange(env.scene.num_envs, device=asset.device, dtype=torch.long)
    else:
        env_ids = env_ids.to(device=asset.device, dtype=torch.long)

    # Keep this as a partial [num_selected_envs, 6] view, matching Isaac Lab's
    # x/y/z/roll/pitch/yaw convention and the Newton/PhysX index-write contract.
    vel_w = asset.data.root_vel_w.torch[env_ids]
    range_list = [velocity_range.get(key, (0.0, 0.0)) for key in ["x", "y", "z", "roll", "pitch", "yaw"]]
    ranges = torch.tensor(range_list, device=asset.device, dtype=vel_w.dtype)
    vel_w += math_utils.sample_uniform(ranges[:, 0], ranges[:, 1], vel_w.shape, device=asset.device)
    asset.write_root_velocity_to_sim_index(root_velocity=vel_w, env_ids=env_ids)


class randomize_joint_default_pos(ManagerTermBase):
    """Randomize joint encoder zeros and matching position-action offsets.

    The unmodified asset defaults are retained as the sampling baseline so a
    reset-mode event resamples calibration error instead of accumulating it.
    """

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg: SceneEntityCfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))
        self.asset: Articulation = env.scene[self.asset_cfg.name]
        self._nominal_default_joint_pos = self.asset.data.default_joint_pos.torch.clone()

    def __call__(
        self,
        env: ManagerBasedRLEnv,
        env_ids: torch.Tensor | None,
        asset_cfg: SceneEntityCfg,
        pos_distribution_params: tuple[float, float],
        operation: Literal["add", "scale", "abs"] = "add",
        distribution: Literal["uniform", "log_uniform", "gaussian"] = "uniform",
        action_name: str = "joint_pos",
    ) -> None:
        selected_envs = (
            torch.arange(env.scene.num_envs, device=self.asset.device, dtype=torch.long)
            if env_ids is None
            else env_ids.to(device=self.asset.device, dtype=torch.long)
        )
        if self.asset_cfg.joint_ids is None or isinstance(self.asset_cfg.joint_ids, slice):
            selected_joints = torch.arange(self.asset.num_joints, device=self.asset.device, dtype=torch.long)
        else:
            selected_joints = torch.as_tensor(self.asset_cfg.joint_ids, device=self.asset.device, dtype=torch.long)

        randomized = isaaclab_events._randomize_prop_by_op(
            self._nominal_default_joint_pos.clone(),
            pos_distribution_params,
            selected_envs,
            selected_joints,
            operation=operation,
            distribution=distribution,
        )
        self.asset.data.default_joint_pos.torch[selected_envs[:, None], selected_joints[None, :]] = randomized[
            selected_envs[:, None], selected_joints[None, :]
        ]

        action_term = env.action_manager.get_term(action_name)
        if not getattr(action_term.cfg, "use_default_offset", False):
            return

        asset_index = {name: index for index, name in enumerate(self.asset.joint_names)}
        selected_names = {self.asset.joint_names[index] for index in selected_joints.tolist()}
        action_indices: list[int] = []
        asset_indices: list[int] = []
        for action_index, name in enumerate(action_term._joint_names):
            if name in selected_names:
                action_indices.append(action_index)
                asset_indices.append(asset_index[name])
        if action_indices:
            action_ids = torch.tensor(action_indices, device=self.asset.device)
            asset_ids = torch.tensor(asset_indices, device=self.asset.device)
            action_term._offset[selected_envs[:, None], action_ids[None, :]] = self.asset.data.default_joint_pos.torch[
                selected_envs[:, None], asset_ids[None, :]
            ]


class actuator_time_delay(ManagerTermBase):
    """Randomize one shared delay step count across an environment's delayed actuators."""

    def __init__(self, cfg: EventTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.asset_cfg = cfg.params.get("asset_cfg", SceneEntityCfg("robot"))

    def __call__(
        self,
        env: ManagerBasedEnv,
        env_ids: torch.Tensor | None,
        delay_steps_range: tuple[int, int],
        asset_cfg: SceneEntityCfg | None = None,
    ) -> None:
        asset_cfg = asset_cfg if asset_cfg is not None else self.asset_cfg
        asset: Articulation = env.scene[asset_cfg.name]
        env_ids_tensor = self._resolve_env_ids(env, env_ids, device=asset.device)
        delayed_actuators = tuple(self._iter_delayed_actuators(asset, asset_cfg))

        if not delayed_actuators:
            raise TypeError("No delayed actuators found for the requested asset configuration.")

        low, high = delay_steps_range
        if low < 0 or high < low:
            raise ValueError(f"delay_steps_range must satisfy 0 <= low <= high, got {delay_steps_range}.")

        common_max_delay = min(
            int(getattr(actuator.cfg, "max_delay", actuator.positions_delay_buffer.history_length))
            for actuator in delayed_actuators
        )
        if high > common_max_delay:
            raise ValueError(
                f"delay_steps_range upper bound {high} exceeds the common actuator max delay {common_max_delay}."
            )

        delay_steps = torch.randint(
            low=low,
            high=high + 1,
            size=(env_ids_tensor.numel(),),
            device=asset.device,
            dtype=torch.int,
        )
        for actuator in delayed_actuators:
            self._set_shared_delay_steps(actuator, delay_steps, env_ids_tensor)

    @staticmethod
    def _resolve_env_ids(env: ManagerBasedEnv, env_ids: torch.Tensor | None, device: torch.device) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(env.scene.num_envs, device=device, dtype=torch.long)
        return env_ids.to(device=device, dtype=torch.long)

    @staticmethod
    def _iter_delayed_actuators(asset: Articulation, asset_cfg: SceneEntityCfg):
        joint_ids = asset_cfg.joint_ids
        if joint_ids is None or isinstance(joint_ids, slice):
            for actuator in asset.actuators.values():
                if hasattr(actuator, "positions_delay_buffer"):
                    yield actuator
            return

        asset_joint_ids = torch.as_tensor(joint_ids, device=asset.device)
        for actuator in asset.actuators.values():
            if not hasattr(actuator, "positions_delay_buffer"):
                continue
            actuator_joint_ids = actuator.joint_indices
            if actuator_joint_ids is None or isinstance(actuator_joint_ids, slice):
                yield actuator
                continue
            actuator_joint_ids = torch.as_tensor(actuator_joint_ids, device=asset.device)
            if torch.isin(actuator_joint_ids, asset_joint_ids).any():
                yield actuator

    @staticmethod
    def _set_shared_delay_steps(actuator, delay_steps: torch.Tensor, env_ids: torch.Tensor) -> None:
        if hasattr(actuator, "set_delay_steps"):
            actuator.set_delay_steps(delay_steps, env_ids=env_ids, reset_buffers=True)
            return
        for buffer_name in ("positions_delay_buffer", "velocities_delay_buffer", "efforts_delay_buffer"):
            delay_buffer = getattr(actuator, buffer_name, None)
            if delay_buffer is None:
                continue
            delay_buffer.set_time_lag(delay_steps, env_ids)
            delay_buffer.reset(env_ids)
