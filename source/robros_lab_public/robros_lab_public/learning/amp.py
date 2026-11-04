# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import inspect
import re
import torch
import torch.nn as nn
from collections.abc import Callable
from tensordict import TensorDict
from torch import autograd
from typing import TYPE_CHECKING, Any

from rsl_rl.env import VecEnv
from rsl_rl.modules import EmpiricalNormalization
from rsl_rl.utils import resolve_callable, resolve_nn_activation, resolve_optimizer

from .amp_motion_loader import AMPMotionLoader, AMPTermSpec
from .replay_buffer import ReplayBuffer

if TYPE_CHECKING:
    from rsl_rl.algorithms.ppo import PPO


def _resolve_base_env(env: VecEnv) -> Any:
    """Return the IsaacLab environment beneath any RL wrapper."""
    return getattr(env, "unwrapped", env)


def _flatten_dim(dim: tuple[int, ...]) -> int:
    total = 1
    for value in dim:
        total *= value
    return total


def _callable_name(func: Callable[..., Any]) -> str:
    return getattr(func, "__name__", func.__class__.__name__)


def _get_callable_default(func: Callable[..., Any], param_name: str) -> Any:
    try:
        parameter = inspect.signature(func).parameters.get(param_name)
    except (TypeError, ValueError):
        return None
    if parameter is None or parameter.default is inspect._empty:
        return None
    return parameter.default


def _get_term_param(term_cfg: Any, param_name: str) -> Any:
    if param_name in term_cfg.params:
        return term_cfg.params[param_name]
    return _get_callable_default(term_cfg.func, param_name)


def _resolve_matching_names(
    selected_names: str | list[str] | tuple[str, ...],
    all_names: list[str],
    preserve_order: bool,
) -> list[str]:
    try:
        import isaaclab.utils.string as string_utils
    except ImportError:
        patterns = [selected_names] if isinstance(selected_names, str) else list(selected_names)
        if preserve_order:
            return [name for name in all_names if any(re.fullmatch(pattern, name) is not None for pattern in patterns)]
        return [name for pattern in patterns for name in all_names if re.fullmatch(pattern, name) is not None]
    _, resolved_names = string_utils.resolve_matching_names(
        selected_names,
        all_names,
        preserve_order=preserve_order,
    )
    return [str(name) for name in resolved_names]


def _resolve_entity_names(base_env: Any, entity_cfg: Any, *, entity_attr: str) -> list[str] | None:
    if entity_cfg is None:
        return None

    entity_name = getattr(entity_cfg, "name", None)
    if entity_name is None:
        return None

    try:
        entity = base_env.scene[entity_name]
    except Exception:
        return None

    if entity_attr == "joint":
        selected_names = getattr(entity_cfg, "joint_names", None)
        selected_ids = getattr(entity_cfg, "joint_ids", slice(None))
        all_names = list(getattr(entity, "joint_names", []))
    else:
        selected_names = getattr(entity_cfg, "body_names", None)
        selected_ids = getattr(entity_cfg, "body_ids", slice(None))
        all_names = list(getattr(entity, "body_names", []))

    preserve_order = bool(getattr(entity_cfg, "preserve_order", False))
    if isinstance(selected_names, (str, list, tuple)):
        return _resolve_matching_names(selected_names, all_names, preserve_order)
    if selected_ids == slice(None):
        return [str(name) for name in all_names]
    if isinstance(selected_ids, int):
        return [str(all_names[selected_ids])]
    return [str(all_names[index]) for index in selected_ids]


def _resolve_anchor_body_name(base_env: Any, anchor_cfg: Any) -> str | None:
    if anchor_cfg is None:
        return None

    body_names = getattr(anchor_cfg, "body_names", None)
    body_ids = getattr(anchor_cfg, "body_ids", slice(None))
    if body_names is None and body_ids == slice(None):
        return None

    resolved_body_names = _resolve_entity_names(base_env, anchor_cfg, entity_attr="body")
    if resolved_body_names is None:
        return None
    if len(resolved_body_names) != 1:
        raise ValueError(
            f"AMP link-position/orientation anchors must resolve to exactly one body, but got {resolved_body_names}."
        )
    return resolved_body_names[0]


def _resolve_amp_term_resolver(term_cfg: Any, base_env: Any) -> dict[str, object] | None:
    func_name = _callable_name(term_cfg.func)

    if func_name in {"joint_pos", "joint_vel"}:
        asset_cfg = _get_term_param(term_cfg, "asset_cfg")
        resolver: dict[str, object] = {"kind": func_name}
        joint_names = _resolve_entity_names(base_env, asset_cfg, entity_attr="joint")
        if joint_names is not None:
            resolver["joint_names"] = joint_names
        return resolver

    if func_name in {"base_lin_vel", "base_ang_vel", "projected_gravity", "link_positions", "link_orientations"}:
        resolver = {"kind": func_name}
        if func_name in {"link_positions", "link_orientations"}:
            asset_cfg = _get_term_param(term_cfg, "asset_cfg")
            body_names = _resolve_entity_names(base_env, asset_cfg, entity_attr="body")
            if body_names is not None:
                resolver["body_names"] = body_names

            anchor_cfg = _get_term_param(term_cfg, "anchor_asset_cfg")
            anchor_body_name = _resolve_anchor_body_name(base_env, anchor_cfg)
            if anchor_body_name is not None:
                resolver["anchor_body_name"] = anchor_body_name
        return resolver

    link_term_kinds = {
        "current_link_height": "link_height",
        "link_lin_vel": "link_lin_vel",
        "link_ang_vel": "link_ang_vel",
        "link_lin_vel_global": "link_lin_vel_global",
        "link_ang_vel_global": "link_ang_vel_global",
        "link_projected_gravity": "link_projected_gravity",
    }
    if func_name in link_term_kinds:
        resolver = {"kind": link_term_kinds[func_name]}
        asset_cfg = _get_term_param(term_cfg, "asset_cfg")
        body_names = _resolve_entity_names(base_env, asset_cfg, entity_attr="body")
        if body_names is not None:
            resolver["body_names"] = body_names
        return resolver

    if func_name in {"current_base_height", "base_pos_z"}:
        return {"kind": "base_height"}

    return None


def _concat_obs_groups(obs: TensorDict, obs_groups: list[str]) -> torch.Tensor:
    obs_list = [obs[obs_group] for obs_group in obs_groups]
    return obs_list[0] if len(obs_list) == 1 else torch.cat(obs_list, dim=-1)


class AMPDiscriminator(nn.Module):
    """Binary expert-vs-policy discriminator used by AMP."""

    def __init__(
        self,
        input_dim: int,
        hidden_dims: tuple[int, ...] | list[int],
        reward_coef: float,
        activation: str = "relu",
    ) -> None:
        """Create the discriminator trunk and scalar prediction head."""
        super().__init__()
        self.input_dim = input_dim
        self.reward_coef = reward_coef

        if len(hidden_dims) == 0:
            raise ValueError("AMPDiscriminator requires at least one hidden layer.")

        layers: list[nn.Module] = []
        in_dim = input_dim
        activation_module = resolve_nn_activation(activation)
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(type(activation_module)())
            in_dim = hidden_dim
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(hidden_dims[-1], 1)

    def forward(self, transitions: torch.Tensor) -> torch.Tensor:
        """Predict the discriminator score for flattened transition windows."""
        return self.head(self.trunk(transitions))

    def compute_grad_pen(
        self,
        expert_transition: torch.Tensor,
        lambda_: float,
    ) -> torch.Tensor:
        """Apply the zero-centered gradient penalty used by AMP."""
        expert_transition = expert_transition.requires_grad_(True)
        logits = self.forward(expert_transition)
        grad_outputs = torch.ones_like(logits)
        grad = autograd.grad(
            outputs=logits,
            inputs=expert_transition,
            grad_outputs=grad_outputs,
            create_graph=True,
            retain_graph=True,
            only_inputs=True,
        )[0]
        return lambda_ * grad.norm(p=2, dim=-1).pow(2).mean()

    def predict_amp_reward(
        self,
        transition: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Convert discriminator outputs into the shaped AMP reward."""
        with torch.no_grad():
            logits = self.forward(transition)
            reward = self.reward_coef * torch.clamp(1.0 - 0.25 * torch.square(logits - 1.0), min=0.0)
        return reward.squeeze(-1), logits


class AMPExtension:
    """AMP discriminator, replay buffer, and reward shaping packaged as a PPO extension.

    Upstream rsl_rl has no extension manager, so this class is standalone and driven by
    :class:`robros_lab_public.learning.amp_ppo.AmpPPO`.
    """

    @classmethod
    def required_observation_sets(cls, cfg: dict) -> tuple[str, ...]:
        """Return discriminator-state and optional motion-label observation sets."""
        obs_sets = [cfg.get("obs_set", "amp")]
        amp_label_obs_set = cfg.get("amp_label_obs_set")
        if amp_label_obs_set is not None:
            obs_sets.append(amp_label_obs_set)
        return tuple(obs_sets)

    @classmethod
    def prepare_config(cls, cfg: dict, obs: TensorDict, obs_groups: dict[str, list[str]], env: VecEnv) -> dict:
        """Resolve AMP terms, timing, labels, and dataset configuration."""
        del obs
        return resolve_amp_config(cfg, env, obs_groups)

    def __init__(self, algorithm: PPO, amp_cfg: dict) -> None:
        """Build expert loaders, replay buffers, discriminators, and optimizers."""
        self.algorithm = algorithm
        self.cfg = dict(amp_cfg)
        self.name = str(self.cfg.get("name", "amp"))
        self.device = algorithm.device
        self.amp_cfg = dict(amp_cfg)
        self.target_reward_group = str(self.amp_cfg.get("target_reward_group", "reward"))
        self.task_reward_lerp = float(self.amp_cfg.get("task_reward_lerp", 0.0))
        if not 0.0 <= self.task_reward_lerp <= 1.0:
            raise ValueError(f"AMP task_reward_lerp must be in [0, 1], got {self.task_reward_lerp}.")
        self.amp_obs_set = str(self.amp_cfg.get("obs_set", "amp"))
        self.amp_obs_groups = list(self.amp_cfg["selected_obs_groups"])
        self.num_transition_steps = int(self.amp_cfg.get("num_transition_steps", 2))
        if self.num_transition_steps < 2:
            raise ValueError(f"AMP num_transition_steps must be at least 2, got {self.num_transition_steps}.")

        discriminator_cfg = dict(self.amp_cfg["discriminator"])
        term_specs = [AMPTermSpec(**spec) if isinstance(spec, dict) else spec for spec in self.amp_cfg["term_specs"]]
        motion_dataset_cfgs = dict(self.amp_cfg["motion_datasets"])
        self.motion_conditioned = len(motion_dataset_cfgs) > 1

        self.amp_label_obs_set = None
        self.amp_label_obs_groups: list[str] = []
        self.amp_label_func = None
        self.amp_label_env = self.amp_cfg.get("_env")
        self.previous_amp_label_obs: torch.Tensor | None = None

        if self.motion_conditioned:
            motion_names = list(self.amp_cfg.get("motion_names") or motion_dataset_cfgs.keys())
            if len(motion_names) != len(motion_dataset_cfgs) or set(motion_names) != set(motion_dataset_cfgs.keys()):
                raise ValueError(
                    "AMP motion_names must contain exactly the keys from motion_datasets, "
                    f"got motion_names={motion_names}, motion_datasets={list(motion_dataset_cfgs.keys())}."
                )
            amp_label_obs_set = self.amp_cfg.get("amp_label_obs_set")
            selected_amp_label_obs_groups = self.amp_cfg.get("selected_amp_label_obs_groups")
            self.amp_label_obs_set = str(amp_label_obs_set) if amp_label_obs_set is not None else None
            self.amp_label_obs_groups = (
                list(selected_amp_label_obs_groups)
                if selected_amp_label_obs_groups is not None
                else ([self.amp_label_obs_set] if self.amp_label_obs_set is not None else [])
            )
            if not self.amp_label_obs_groups:
                raise ValueError("AMP motion_datasets require labeling via 'amp_label_obs_set'.")
            amp_label_func = self.amp_cfg.get("amp_label_func")
            if amp_label_func is None:
                raise ValueError("AMP motion_datasets require 'amp_label_func' to route labels to discriminators.")
            self.amp_label_func = resolve_callable(amp_label_func)
            if not callable(self.amp_label_func):
                raise TypeError(f"AMP amp_label_func must be callable, got {self.amp_label_func}.")

            self.motion_names = tuple(str(name) for name in motion_names)
            self.motion_amp_data: dict[str, AMPMotionLoader] = {}
            for motion_name in self.motion_names:
                self.motion_amp_data[motion_name] = AMPMotionLoader(
                    term_specs=term_specs,
                    time_between_frames=float(self.amp_cfg["time_between_frames"]),
                    device=self.device,
                    num_transition_steps=self.num_transition_steps,
                    **dict(motion_dataset_cfgs[motion_name]),
                )
            self.amp_data = self.motion_amp_data[self.motion_names[0]]
            for motion_name, amp_data in self.motion_amp_data.items():
                if amp_data.observation_dim != self.amp_data.observation_dim:
                    raise ValueError(
                        f"AMP motion dataset '{motion_name}' observation dim {amp_data.observation_dim} "
                        f"does not match {self.amp_data.observation_dim}."
                    )
                if amp_data.transition_dim != self.amp_data.transition_dim:
                    raise ValueError(
                        f"AMP motion dataset '{motion_name}' transition dim {amp_data.transition_dim} "
                        f"does not match {self.amp_data.transition_dim}."
                    )
        else:
            self.motion_names = ()
            self.motion_amp_data = {}
            dataset_cfg = dict(next(iter(motion_dataset_cfgs.values())))
            self.amp_data = AMPMotionLoader(
                term_specs=term_specs,
                time_between_frames=float(self.amp_cfg["time_between_frames"]),
                device=self.device,
                num_transition_steps=self.num_transition_steps,
                **dataset_cfg,
            )

        discriminator_input_dim = self.amp_data.transition_dim

        self.amp_replay_buffer = ReplayBuffer(
            obs_dim=discriminator_input_dim,
            buffer_size=int(self.amp_cfg.get("replay_buffer_size", 100000)),
            device=self.device,
        )
        self.amp_replay_buffers = (
            {
                motion_name: ReplayBuffer(
                    obs_dim=discriminator_input_dim,
                    buffer_size=int(self.amp_cfg.get("replay_buffer_size", 100000)),
                    device=self.device,
                )
                for motion_name in self.motion_names
            }
            if self.motion_conditioned
            else {}
        )

        state_normalization = bool(self.amp_cfg.get("state_normalization", True))
        normalization_until = self.amp_cfg.get("normalization_until", int(1.0e8))
        self.amp_state_normalizer = (
            EmpiricalNormalization(self.amp_data.observation_dim, until=normalization_until).to(self.device)
            if state_normalization
            else None
        )

        self.amp_discriminator_hidden_dims = discriminator_cfg.pop("hidden_dims")
        self.amp_reward_coef = float(discriminator_cfg.pop("reward_coef"))
        self.amp_grad_penalty_coef = float(discriminator_cfg.pop("grad_penalty_coef", 10.0))
        self.amp_max_grad_norm = discriminator_cfg.pop("max_grad_norm", None)
        self.amp_optimizer_name = discriminator_cfg.pop("optimizer", "adam")
        amp_learning_rate = discriminator_cfg.pop("learning_rate", None)
        self.amp_learning_rate = algorithm.learning_rate if amp_learning_rate is None else float(amp_learning_rate)
        self.amp_trunk_weight_decay = float(discriminator_cfg.pop("trunk_weight_decay", 1.0e-3))
        self.amp_head_weight_decay = float(discriminator_cfg.pop("head_weight_decay", 1.0e-1))
        self.amp_discriminator_kwargs = discriminator_cfg

        if self.motion_conditioned:
            self.discriminators = nn.ModuleDict()
            self.discriminator_optimizers = {}
            for motion_name in self.motion_names:
                discriminator, optimizer = self._build_discriminator_and_optimizer(discriminator_input_dim)
                self.discriminators[motion_name] = discriminator
                self.discriminator_optimizers[motion_name] = optimizer
        else:
            self.discriminators = nn.ModuleDict()
            discriminator, optimizer = self._build_discriminator_and_optimizer(discriminator_input_dim)
            self.discriminators["default"] = discriminator
            self.discriminator_optimizers = {"default": optimizer}

        self.previous_obs: torch.Tensor | None = None
        self.policy_obs_history: torch.Tensor | None = None
        self.policy_obs_history_lengths: torch.Tensor | None = None
        self._rollout_amp_reward_sum = 0.0
        self._rollout_amp_reward_count = 0

    def _build_discriminator_and_optimizer(
        self, discriminator_input_dim: int
    ) -> tuple[AMPDiscriminator, torch.optim.Optimizer]:
        discriminator = AMPDiscriminator(
            input_dim=discriminator_input_dim,
            hidden_dims=self.amp_discriminator_hidden_dims,
            reward_coef=self.amp_reward_coef,
            **self.amp_discriminator_kwargs,
        ).to(self.device)
        optimizer = resolve_optimizer(self.amp_optimizer_name)(
            [
                {
                    "params": discriminator.trunk.parameters(),
                    "weight_decay": self.amp_trunk_weight_decay,
                    "name": "trunk",
                },
                {
                    "params": discriminator.head.parameters(),
                    "weight_decay": self.amp_head_weight_decay,
                    "name": "head",
                },
            ],
            lr=self.amp_learning_rate,
        )
        return discriminator, optimizer

    @property
    def discriminator(self) -> AMPDiscriminator:
        """Return the single or first motion-conditioned discriminator."""
        key = self.motion_names[0] if self.motion_conditioned else "default"
        return self.discriminators[key]

    @property
    def optimizer(self) -> torch.optim.Optimizer:
        """Return the single or first motion-conditioned optimizer."""
        key = self.motion_names[0] if self.motion_conditioned else "default"
        return self.discriminator_optimizers[key]

    def on_rollout_action(self, obs: TensorDict) -> None:
        """Record the current AMP observation before the policy acts."""
        self.previous_obs = self._extract_amp_obs(obs).detach().clone()
        if self.motion_conditioned:
            self.previous_amp_label_obs = self._extract_amp_label_obs(obs).detach().clone()
        self._append_policy_obs(self.previous_obs)

    def update_reward_groups(
        self,
        obs: TensorDict,
        reward_groups: dict[str, torch.Tensor],
        dones: torch.Tensor,
        extras: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """Replace environment rewards with AMP-shaped rewards and append replay transitions."""
        task_rewards = reward_groups[self.target_reward_group].view(-1)
        if self.previous_obs is None:
            raise RuntimeError("AMPExtension requires on_rollout_action() before update_reward_groups().")

        next_amp_obs = self._extract_amp_obs(obs).detach()
        amp_extras = extras.get("amp", {})
        terminal_amp_obs = amp_extras.get("terminal_observation")
        done_mask = dones.view(-1).bool()
        if terminal_amp_obs is not None:
            next_amp_obs = next_amp_obs.clone()
            next_amp_obs[done_mask] = terminal_amp_obs.to(self.device)[done_mask]
        elif torch.any(done_mask):
            raise KeyError(
                "AMP requires extras['amp']['terminal_observation'] for terminated environments."
                " Use the AMP-specific manager-based env entry point."
            )

        policy_transition = self._build_policy_transition(next_amp_obs)
        if self.motion_conditioned:
            if self.previous_amp_label_obs is None:
                raise RuntimeError("Motion-conditioned AMP requires on_rollout_action() to record the AMP label.")
            motion_indices = self._select_motion_indices(self.previous_amp_label_obs)
            amp_rewards = torch.zeros_like(task_rewards)
            for motion_index, motion_name in enumerate(self.motion_names):
                motion_mask = motion_indices == motion_index
                if not torch.any(motion_mask):
                    continue
                normalized_policy_transition = self._normalize_transition(policy_transition[motion_mask])
                motion_rewards, _ = self.discriminators[motion_name].predict_amp_reward(
                    normalized_policy_transition,
                )
                amp_rewards[motion_mask] = motion_rewards
        else:
            normalized_policy_transition = self._normalize_transition(policy_transition)
            amp_rewards, _ = self.discriminator.predict_amp_reward(normalized_policy_transition)

        if self.policy_obs_history_lengths is None:
            raise RuntimeError("AMP policy observation history is not initialized.")
        full_history_mask = self.policy_obs_history_lengths >= self.num_transition_steps - 1
        if torch.any(full_history_mask):
            if self.motion_conditioned:
                for motion_index, motion_name in enumerate(self.motion_names):
                    motion_mask = full_history_mask & (motion_indices == motion_index)
                    if torch.any(motion_mask):
                        self.amp_replay_buffers[motion_name].insert(policy_transition[motion_mask])
            else:
                self.amp_replay_buffer.insert(policy_transition[full_history_mask])
        self._rollout_amp_reward_sum += amp_rewards.sum().item()
        self._rollout_amp_reward_count += amp_rewards.numel()
        if torch.any(done_mask):
            self._reset_policy_history(done_mask)
        self.previous_obs = None
        self.previous_amp_label_obs = None
        blended_rewards = self.task_reward_lerp * task_rewards + (1.0 - self.task_reward_lerp) * amp_rewards
        return {self.target_reward_group: blended_rewards.view_as(reward_groups[self.target_reward_group])}

    def run_auxiliary_update(self) -> dict[str, float]:
        """Train the discriminator on expert and policy transitions collected so far."""
        num_updates = self.algorithm.num_learning_epochs * self.algorithm.num_mini_batches
        mini_batch_size = max(
            1,
            self.algorithm.storage.num_envs
            * self.algorithm.storage.num_transitions_per_env
            // self.algorithm.num_mini_batches,
        )
        if self.motion_conditioned:
            loss_dict = self._update_motion_discriminators(num_updates, mini_batch_size)
            if self._rollout_amp_reward_count > 0:
                loss_dict["reward"] = self._rollout_amp_reward_sum / self._rollout_amp_reward_count
            self._rollout_amp_reward_sum = 0.0
            self._rollout_amp_reward_count = 0
            return loss_dict

        mean_amp_loss = 0.0
        mean_grad_penalty_loss = 0.0
        mean_policy_pred = 0.0
        mean_expert_pred = 0.0

        expert_generator = self.amp_data.feed_forward_generator(num_updates, mini_batch_size)
        policy_generator = self.amp_replay_buffer.feed_forward_generator(num_updates, mini_batch_size)

        for policy_transition in policy_generator:
            expert_transition = next(expert_generator)

            raw_expert_transition = expert_transition
            raw_policy_transition = policy_transition

            if self.amp_state_normalizer is not None:
                with torch.no_grad():
                    expert_transition = self._normalize_transition(expert_transition)
                    policy_transition = self._normalize_transition(policy_transition)

            expert_pred = self.discriminator(expert_transition)
            policy_pred = self.discriminator(policy_transition)

            expert_loss = nn.functional.mse_loss(expert_pred, torch.ones_like(expert_pred))
            policy_loss = nn.functional.mse_loss(policy_pred, -torch.ones_like(policy_pred))
            amp_loss = 0.5 * (expert_loss + policy_loss)
            grad_penalty_loss = self.discriminator.compute_grad_pen(
                expert_transition,
                lambda_=self.amp_grad_penalty_coef,
            )
            total_loss = amp_loss + grad_penalty_loss

            self.optimizer.zero_grad(set_to_none=True)
            total_loss.backward()
            if self.algorithm.is_multi_gpu:
                self.algorithm._reduce_gradients(self.discriminator.parameters())
            if self.amp_max_grad_norm is not None:
                nn.utils.clip_grad_norm_(self.discriminator.parameters(), self.amp_max_grad_norm)
            self.optimizer.step()

            if self.amp_state_normalizer is not None:
                self._update_transition_normalizer(raw_expert_transition)
                self._update_transition_normalizer(raw_policy_transition)

            mean_amp_loss += amp_loss.item()
            mean_grad_penalty_loss += grad_penalty_loss.item()
            mean_policy_pred += policy_pred.mean().item()
            mean_expert_pred += expert_pred.mean().item()

        loss_dict = {
            "discriminator": mean_amp_loss / num_updates,
            "gradient_penalty": mean_grad_penalty_loss / num_updates,
            "policy_prediction": mean_policy_pred / num_updates,
            "expert_prediction": mean_expert_pred / num_updates,
        }
        if self._rollout_amp_reward_count > 0:
            loss_dict["reward"] = self._rollout_amp_reward_sum / self._rollout_amp_reward_count

        self._rollout_amp_reward_sum = 0.0
        self._rollout_amp_reward_count = 0
        return loss_dict

    def _update_motion_discriminators(self, num_updates: int, mini_batch_size: int) -> dict[str, float]:
        loss_dict: dict[str, float] = {}
        total_amp_loss = 0.0
        total_grad_penalty_loss = 0.0
        total_policy_pred = 0.0
        total_expert_pred = 0.0
        total_count = 0

        for motion_name in self.motion_names:
            replay_buffer = self.amp_replay_buffers[motion_name]
            if replay_buffer.num_samples == 0:
                continue

            discriminator = self.discriminators[motion_name]
            optimizer = self.discriminator_optimizers[motion_name]
            expert_generator = self.motion_amp_data[motion_name].feed_forward_generator(num_updates, mini_batch_size)
            policy_generator = replay_buffer.feed_forward_generator(num_updates, mini_batch_size)

            motion_amp_loss = 0.0
            motion_grad_penalty_loss = 0.0
            motion_policy_pred = 0.0
            motion_expert_pred = 0.0
            motion_count = 0

            for policy_transition in policy_generator:
                expert_transition = next(expert_generator)
                raw_expert_transition = expert_transition
                raw_policy_transition = policy_transition

                if self.amp_state_normalizer is not None:
                    with torch.no_grad():
                        expert_transition = self._normalize_transition(expert_transition)
                        policy_transition = self._normalize_transition(policy_transition)

                expert_pred = discriminator(expert_transition)
                policy_pred = discriminator(policy_transition)
                expert_loss = nn.functional.mse_loss(expert_pred, torch.ones_like(expert_pred))
                policy_loss = nn.functional.mse_loss(policy_pred, -torch.ones_like(policy_pred))
                amp_loss = 0.5 * (expert_loss + policy_loss)
                grad_penalty_loss = discriminator.compute_grad_pen(
                    expert_transition,
                    lambda_=self.amp_grad_penalty_coef,
                )
                total_loss = amp_loss + grad_penalty_loss

                optimizer.zero_grad(set_to_none=True)
                total_loss.backward()
                if self.algorithm.is_multi_gpu:
                    self.algorithm._reduce_gradients(discriminator.parameters())
                if self.amp_max_grad_norm is not None:
                    nn.utils.clip_grad_norm_(discriminator.parameters(), self.amp_max_grad_norm)
                optimizer.step()

                if self.amp_state_normalizer is not None:
                    self._update_transition_normalizer(raw_expert_transition)
                    self._update_transition_normalizer(raw_policy_transition)

                motion_amp_loss += amp_loss.item()
                motion_grad_penalty_loss += grad_penalty_loss.item()
                motion_policy_pred += policy_pred.mean().item()
                motion_expert_pred += expert_pred.mean().item()
                motion_count += 1

            if motion_count == 0:
                continue
            loss_dict[f"discriminator/{motion_name}"] = motion_amp_loss / motion_count
            loss_dict[f"gradient_penalty/{motion_name}"] = motion_grad_penalty_loss / motion_count
            loss_dict[f"policy_prediction/{motion_name}"] = motion_policy_pred / motion_count
            loss_dict[f"expert_prediction/{motion_name}"] = motion_expert_pred / motion_count

            total_amp_loss += motion_amp_loss
            total_grad_penalty_loss += motion_grad_penalty_loss
            total_policy_pred += motion_policy_pred
            total_expert_pred += motion_expert_pred
            total_count += motion_count

        if total_count == 0:
            return {
                "discriminator": 0.0,
                "gradient_penalty": 0.0,
                "policy_prediction": 0.0,
                "expert_prediction": 0.0,
            }

        loss_dict.update({
            "discriminator": total_amp_loss / total_count,
            "gradient_penalty": total_grad_penalty_loss / total_count,
            "policy_prediction": total_policy_pred / total_count,
            "expert_prediction": total_expert_pred / total_count,
        })
        return loss_dict

    def train_mode(self) -> None:
        """Enable training mode so the AMP state normalizer accepts updates."""
        self.discriminators.train()
        if self.amp_state_normalizer is not None:
            self.amp_state_normalizer.train()

    def eval_mode(self) -> None:
        """Freeze the discriminators and the AMP state normalizer."""
        self.discriminators.eval()
        if self.amp_state_normalizer is not None:
            self.amp_state_normalizer.eval()

    def checkpoint_state(self) -> dict[str, Any]:
        """Return discriminator, optimizer and normalizer state for checkpoints."""
        state: dict[str, Any] = {
            "discriminators": self.discriminators.state_dict(),
            "optimizers": {name: opt.state_dict() for name, opt in self.discriminator_optimizers.items()},
        }
        if self.amp_state_normalizer is not None:
            state["normalizer"] = self.amp_state_normalizer.state_dict()
        return state

    def load_checkpoint_state(self, state: dict[str, Any], load_optimizer: bool, strict: bool) -> None:
        """Restore state written by :meth:`checkpoint_state`."""
        self.discriminators.load_state_dict(state["discriminators"], strict=strict)
        if load_optimizer:
            for name, opt in self.discriminator_optimizers.items():
                opt.load_state_dict(state["optimizers"][name])
        if self.amp_state_normalizer is not None and "normalizer" in state:
            self.amp_state_normalizer.load_state_dict(state["normalizer"], strict=strict)

    def auxiliary_optimizers(self) -> dict[str, torch.optim.Optimizer]:
        """Expose discriminator optimizers stepped by AMP's autonomous update loop."""
        return {
            self._optimizer_name(discriminator_name): optimizer
            for discriminator_name, optimizer in self.discriminator_optimizers.items()
        }

    def _optimizer_name(self, discriminator_name: str) -> str:
        if self.motion_conditioned:
            return f"{self.name}/discriminator/{discriminator_name}"
        return f"{self.name}/discriminator"

    def _append_policy_obs(self, amp_obs: torch.Tensor) -> None:
        """Append the current policy AMP observation to the rolling per-env history."""
        num_envs, obs_dim = amp_obs.shape
        history_steps = self.num_transition_steps - 1
        if self.policy_obs_history is None:
            self.policy_obs_history = torch.zeros(num_envs, history_steps, obs_dim, device=self.device)
            self.policy_obs_history_lengths = torch.zeros(num_envs, dtype=torch.long, device=self.device)
        elif self.policy_obs_history.shape != (num_envs, history_steps, obs_dim):
            raise ValueError(
                "AMP policy observation shape changed during rollout: "
                f"expected {(num_envs, history_steps, obs_dim)}, got {self.policy_obs_history.shape}."
            )

        self.policy_obs_history = torch.roll(self.policy_obs_history, shifts=-1, dims=1)
        self.policy_obs_history[:, -1] = amp_obs
        if self.policy_obs_history_lengths is None:
            raise RuntimeError("AMP policy observation history lengths are not initialized.")
        self.policy_obs_history_lengths.add_(1).clamp_(max=history_steps)

    def _build_policy_transition(self, next_amp_obs: torch.Tensor) -> torch.Tensor:
        """Build a flattened transition window ending at ``next_amp_obs``."""
        if self.policy_obs_history is None or self.policy_obs_history_lengths is None:
            raise RuntimeError("AMPExtension requires on_rollout_action() before on_environment_step().")

        frames = torch.empty(
            next_amp_obs.shape[0],
            self.num_transition_steps,
            self.amp_data.observation_dim,
            device=self.device,
            dtype=next_amp_obs.dtype,
        )
        frames[:, :-1] = self.policy_obs_history
        frames[:, -1] = next_amp_obs

        valid_start_indices = (self.num_transition_steps - 1 - self.policy_obs_history_lengths).clamp(min=0)
        for start_index in torch.unique(valid_start_indices).tolist():
            if start_index <= 0:
                continue
            mask = valid_start_indices == start_index
            frames[mask, :start_index] = frames[mask, start_index : start_index + 1]

        return frames.reshape(next_amp_obs.shape[0], -1)

    def _reset_policy_history(self, done_mask: torch.Tensor) -> None:
        """Clear AMP observation history for environments that just reset."""
        if self.policy_obs_history is None or self.policy_obs_history_lengths is None:
            return
        self.policy_obs_history[done_mask] = 0.0
        self.policy_obs_history_lengths[done_mask] = 0

    def _transition_frames(self, transition: torch.Tensor) -> torch.Tensor:
        return transition.reshape(-1, self.num_transition_steps, self.amp_data.observation_dim)

    def _normalize_transition(self, transition: torch.Tensor) -> torch.Tensor:
        if self.amp_state_normalizer is None:
            return transition
        frames = self._transition_frames(transition)
        normalized_frames = self.amp_state_normalizer(frames.reshape(-1, self.amp_data.observation_dim))
        return normalized_frames.reshape(transition.shape[0], -1)

    def _update_transition_normalizer(self, transition: torch.Tensor) -> None:
        if self.amp_state_normalizer is None:
            return
        frames = self._transition_frames(transition)
        self.amp_state_normalizer.update(frames.reshape(-1, self.amp_data.observation_dim))

    def _extract_amp_obs(self, obs: TensorDict) -> torch.Tensor:
        return _concat_obs_groups(obs, self.amp_obs_groups)

    def _extract_amp_label_obs(self, obs: TensorDict) -> torch.Tensor:
        for obs_group in self.amp_label_obs_groups:
            if obs_group not in obs:
                raise KeyError(
                    f"Motion-conditioned AMP label observation group '{obs_group}' is not present in obs. "
                    f"Available groups: {list(obs.keys())}"
                )
        label_obs = _concat_obs_groups(obs, self.amp_label_obs_groups)
        if label_obs.ndim != 2:
            raise ValueError(f"Motion-conditioned AMP label observation must be 2D, got shape {label_obs.shape}.")
        return label_obs

    def _select_motion_indices(self, label_obs: torch.Tensor) -> torch.Tensor:
        if self.amp_label_func is None:
            raise RuntimeError("Motion-conditioned AMP requires an amp_label_func.")

        motion_indices = self.amp_label_func(
            label_obs=label_obs,
            motion_names=self.motion_names,
            env=self.amp_label_env,
        )
        if not torch.is_tensor(motion_indices):
            motion_indices = torch.as_tensor(motion_indices, device=self.device)
        else:
            motion_indices = motion_indices.to(device=self.device)
        if motion_indices.ndim != 1 or motion_indices.shape[0] != label_obs.shape[0]:
            raise ValueError(
                "AMP amp_label_func must return one discriminator index per environment, "
                f"got shape {tuple(motion_indices.shape)} for label batch {label_obs.shape[0]}."
            )
        if motion_indices.dtype.is_floating_point:
            rounded_indices = torch.round(motion_indices)
            if not torch.allclose(motion_indices, rounded_indices, atol=1.0e-4, rtol=0.0):
                invalid_values = motion_indices.detach().cpu().tolist()
                raise ValueError(
                    f"AMP amp_label_func returned non-integer discriminator indices, got {invalid_values}."
                )
            motion_indices = rounded_indices
        motion_indices = motion_indices.to(dtype=torch.long)
        invalid_mask = (motion_indices < 0) | (motion_indices >= len(self.motion_names))
        if torch.any(invalid_mask):
            invalid_values = motion_indices[invalid_mask].detach().cpu().tolist()
            raise ValueError(
                "AMP amp_label_func returned discriminator indices outside "
                f"[0, {len(self.motion_names) - 1}], got {invalid_values}."
            )
        return motion_indices


def resolve_amp_config(amp_cfg: dict, env: VecEnv, obs_groups: dict[str, list[str]]) -> dict:
    """Resolve AMP observation metadata from IsaacLab's observation manager."""
    amp_cfg = dict(amp_cfg)

    base_env = _resolve_base_env(env)

    obs_set = amp_cfg.get("obs_set", "amp")
    if obs_set not in obs_groups:
        raise KeyError(
            f"AMP requires an '{obs_set}' entry in 'obs_groups'. Available observation sets: {list(obs_groups.keys())}"
        )

    observation_manager = getattr(base_env, "observation_manager", None)
    if observation_manager is None:
        raise AttributeError("AMP requires access to IsaacLab's observation_manager to resolve term metadata.")

    group_term_cfgs = getattr(observation_manager, "_group_obs_term_cfgs", {})
    term_specs: list[dict[str, object]] = []
    selected_groups = obs_groups[obs_set]

    for group_name in selected_groups:
        term_names = observation_manager.active_terms[group_name]
        term_dims = observation_manager.group_obs_term_dim[group_name]
        term_cfgs = group_term_cfgs.get(group_name, [])
        if len(term_names) != len(term_dims) or (term_cfgs and len(term_cfgs) != len(term_names)):
            raise ValueError(f"AMP observation metadata for group '{group_name}' is inconsistent.")
        for index, (term_name, term_dim) in enumerate(zip(term_names, term_dims)):
            if term_cfgs:
                history_length = int(getattr(term_cfgs[index], "history_length", 0))
                if history_length > 0:
                    raise ValueError(
                        f"AMP observation term '{group_name}.{term_name}' uses history_length={history_length}."
                        " AMP observation groups must be history-free because the discriminator already trains on"
                        " state transitions."
                    )
            term_spec = {"group_name": group_name, "name": term_name, "dim": _flatten_dim(term_dim)}
            if term_cfgs:
                resolver = _resolve_amp_term_resolver(term_cfgs[index], base_env)
                if resolver is not None:
                    term_spec["resolver"] = resolver
            term_specs.append(term_spec)

    amp_cfg["selected_obs_groups"] = selected_groups
    amp_cfg["term_specs"] = term_specs
    amp_cfg["num_transition_steps"] = int(amp_cfg.get("num_transition_steps", 2))
    if amp_cfg["num_transition_steps"] < 2:
        raise ValueError(f"AMP num_transition_steps must be at least 2, got {amp_cfg['num_transition_steps']}.")
    num_motion_bins = int(amp_cfg.get("num_motion_bins", 1))
    if num_motion_bins > 1:
        raise NotImplementedError("AMP num_motion_bins was removed. Use motion_datasets with amp_label_obs_set.")

    motion_dataset_cfgs = dict(amp_cfg.get("motion_datasets") or {})
    if not motion_dataset_cfgs:
        raise KeyError("AMPExtension requires at least one entry in 'motion_datasets'.")

    for motion_name, motion_dataset_cfg in motion_dataset_cfgs.items():
        if dict(motion_dataset_cfg).get("motion_dir") is None:
            raise KeyError(f"AMP motion_datasets['{motion_name}'] requires a motion_dir value.")

    if len(motion_dataset_cfgs) > 1:
        amp_label_obs_set = amp_cfg.get("amp_label_obs_set")
        if amp_label_obs_set is None:
            raise ValueError("AMP motion_datasets require labeling via 'amp_label_obs_set'.")
        if amp_cfg.get("amp_label_func") is None:
            raise ValueError("AMP motion_datasets require 'amp_label_func' to route labels to discriminators.")
        if amp_label_obs_set not in obs_groups:
            raise KeyError(
                f"AMP amp_label_obs_set '{amp_label_obs_set}' is not present in obs_groups. "
                f"Available observation sets: {list(obs_groups.keys())}"
            )

        motion_names = list(amp_cfg.get("motion_names") or motion_dataset_cfgs.keys())
        if len(motion_names) != len(motion_dataset_cfgs) or set(motion_names) != set(motion_dataset_cfgs.keys()):
            raise ValueError(
                "AMP motion_names must contain exactly the keys from motion_datasets, "
                f"got motion_names={motion_names}, motion_datasets={list(motion_dataset_cfgs.keys())}."
            )
        amp_cfg["motion_names"] = motion_names
        amp_cfg["amp_label_obs_set"] = amp_label_obs_set
        amp_cfg["selected_amp_label_obs_groups"] = obs_groups[amp_label_obs_set]
        # The label callback may need Isaac Lab managers (for example,
        # ``command_manager``), which are not exposed by the RSL-RL wrapper.
        # Keep the unwrapped environment resolved above instead of ``env``.
        amp_cfg["_env"] = base_env
    amp_cfg["motion_datasets"] = motion_dataset_cfgs

    if amp_cfg.get("time_between_frames") is None:
        step_dt = getattr(base_env, "step_dt", None)
        if step_dt is None:
            raise AttributeError("AMP requires 'step_dt' on the environment to resolve transition timing.")
        amp_cfg["time_between_frames"] = float(step_dt)

    return amp_cfg
