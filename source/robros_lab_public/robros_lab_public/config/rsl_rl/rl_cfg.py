from __future__ import annotations

from collections.abc import Callable
from dataclasses import MISSING
from typing import Literal

from isaaclab.utils.configclass import configclass
from isaaclab_rl.rsl_rl import RslRlMLPModelCfg
from isaaclab_rl.rsl_rl import RslRlPpoAlgorithmCfg as IsaacLabRslRlPpoAlgorithmCfg


@configclass
class RslRlCustomMLPModelCfg(RslRlMLPModelCfg):
    """Configuration for an MLP with per-observation-group normalization."""

    class_name: str = "robros_lab_public.learning.custom_mlp_model:CustomMLPModel"
    """The model class name."""

    normalized_obs_groups: list[str] | None = None
    """Active observation groups normalized by the model's empirical normalizer.

    If ``None``, ``obs_normalization=True`` normalizes all active groups, matching ``MLPModel``. An empty list disables
    observation normalization while still allowing the model to consume multiple input groups.
    """


@configclass
class RslRlAmpMotionDatasetCfg:
    """Configuration of the offline expert dataset used by AMP."""

    motion_dir: str | None = None
    """Directory containing canonical expert-motion ``.pkl`` clips.

    Root-level PKLs form one discriminator. If the directory contains subdirectories, each subdirectory forms one
    discriminator.
    """

    mjcf_file: str | None = None
    """MJCF used to derive AMP body observations from canonical root/DoF data."""

    preload_transitions: bool = True
    """Whether to preload expert transitions into GPU memory."""

    num_preload_transitions: int = 200000
    """Number of expert transitions to preload when ``preload_transitions`` is enabled."""


@configclass
class RslRlAmpDiscriminatorCfg:
    """Configuration of the AMP discriminator network and optimizer."""

    hidden_dims: list[int] = MISSING
    """Hidden dimensions of the discriminator MLP."""

    reward_coef: float = 2.0
    """Overall scale applied to the discriminator reward."""

    activation: str = "relu"
    """Activation function of the discriminator MLP."""

    grad_penalty_coef: float = 10.0
    """Coefficient of the zero-centered discriminator gradient penalty."""

    optimizer: Literal["adam", "adamw", "sgd", "rmsprop"] = "adam"
    """Optimizer used for the discriminator update."""

    learning_rate: float | None = None
    """Learning rate of the discriminator optimizer. ``None`` reuses the PPO learning rate."""

    max_grad_norm: float | None = None
    """Optional discriminator gradient clipping norm."""

    trunk_weight_decay: float = 1.0e-3
    """Weight decay applied to the discriminator trunk."""

    head_weight_decay: float = 1.0e-1
    """Weight decay applied to the discriminator head."""


@configclass
class RslRlAmpCfg:
    """Configuration of the AMP-specific training components."""

    name: str = "amp"
    """Name used as the AMP logging prefix and checkpoint key."""

    target_reward_group: str = "reward"
    """Reward group replaced by the task/AMP blend."""

    task_reward_lerp: float = 0.0
    """Blend used as ``task_reward_lerp * task_reward + (1 - task_reward_lerp) * amp_reward``."""

    obs_set: str = "amp"
    """Observation set name used to assemble discriminator inputs from the environment."""

    replay_buffer_size: int = 100000
    """Number of policy transitions retained for discriminator training."""

    state_normalization: bool = True
    """Whether to normalize AMP states with running empirical statistics."""

    normalization_until: int | None = int(1.0e8)
    """Maximum number of AMP samples used to fit the state normalizer."""

    time_between_frames: float | None = None
    """Spacing in seconds between consecutive AMP transition observations. ``None`` defaults to ``env.step_dt``."""

    num_transition_steps: int = 2
    """Number of consecutive AMP observations concatenated into each discriminator transition."""

    motion_names: list[str] = []
    """Optional discriminator order for multi-entry AMP motion datasets."""

    motion_datasets: dict[str, RslRlAmpMotionDatasetCfg] = MISSING
    """Discriminator-name to expert-motion dataset mapping.

    A one-entry mapping configures standard single-discriminator AMP. Multiple
    entries configure motion-conditioned AMP with one discriminator per entry.
    """

    amp_label_obs_set: str | None = None
    """Observation set used to select a discriminator for multi-dataset AMP."""

    amp_label_func: Callable | str | None = None
    """Function that maps AMP-label observations to discriminator indices.

    Called as ``amp_label_func(label_obs=..., motion_names=..., env=...)`` and expected to return one integer
    discriminator index per environment.
    """

    discriminator: RslRlAmpDiscriminatorCfg = MISSING
    """Discriminator and reward-shaping configuration."""


@configclass
class RslRlAmpPpoAlgorithmCfg(IsaacLabRslRlPpoAlgorithmCfg):
    """Upstream rsl_rl PPO (``symmetry_cfg``) with the AMP extension."""

    class_name: str = "robros_lab_public.learning.amp_ppo:AmpPPO"
    """The algorithm class name."""

    amp_cfg: RslRlAmpCfg = MISSING
    """AMP configuration, consumed by :class:`robros_lab_public.learning.amp.AMPExtension`."""
