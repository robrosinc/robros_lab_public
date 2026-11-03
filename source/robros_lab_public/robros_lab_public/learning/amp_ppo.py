from __future__ import annotations

import torch
from tensordict import TensorDict

from rsl_rl.algorithms import PPO
from rsl_rl.env import VecEnv

from .amp import AMPExtension, resolve_amp_config


class AmpPPO(PPO):
    """Upstream rsl_rl PPO driving :class:`AMPExtension`.

    The extension is called at the following points:

    * :meth:`act`: record the AMP observation before the policy acts.
    * :meth:`process_env_step`: blend the AMP reward into the task reward before time-out bootstrapping.
    * :meth:`update`: train the discriminator after the PPO update.
    """

    def __init__(self, actor, critic, storage, amp_cfg: dict | None = None, **kwargs) -> None:
        super().__init__(actor, critic, storage, **kwargs)
        if amp_cfg is None:
            raise ValueError("AmpPPO requires 'amp_cfg' in the algorithm configuration.")
        if self.is_multi_gpu:
            raise NotImplementedError("AmpPPO does not support multi-GPU training.")
        self.amp = AMPExtension(self, amp_cfg)

    @staticmethod
    def construct_algorithm(obs: TensorDict, env: VecEnv, cfg: dict, device: str) -> PPO:
        """Resolve the AMP term metadata from the environment, then build the algorithm like PPO."""
        cfg["algorithm"]["amp_cfg"] = resolve_amp_config(dict(cfg["algorithm"]["amp_cfg"]), env, cfg["obs_groups"])
        return PPO.construct_algorithm(obs, env, cfg, device)

    def act(self, obs: TensorDict) -> torch.Tensor:
        self.amp.on_rollout_action(obs)
        return super().act(obs)

    def process_env_step(
        self, obs: TensorDict, rewards: torch.Tensor, dones: torch.Tensor, extras: dict[str, torch.Tensor]
    ) -> None:
        reward_groups = self.amp.update_reward_groups(obs, {self.amp.target_reward_group: rewards}, dones, extras)
        super().process_env_step(obs, reward_groups[self.amp.target_reward_group], dones, extras)

    def update(self) -> dict[str, float]:
        loss_dict = super().update()
        # AMP metrics use "<name>/<metric>" keys (logged as Loss/amp/...).
        loss_dict.update({f"{self.amp.name}/{key}": value for key, value in self.amp.run_auxiliary_update().items()})
        return loss_dict

    def train_mode(self) -> None:
        super().train_mode()
        self.amp.train_mode()

    def eval_mode(self) -> None:
        super().eval_mode()
        self.amp.eval_mode()

    def save(self) -> dict:
        saved_dict = super().save()
        saved_dict["amp_state"] = self.amp.checkpoint_state()
        return saved_dict

    def load(self, loaded_dict: dict, load_cfg: dict | None, strict: bool) -> bool:
        load_iteration = super().load(loaded_dict, load_cfg, strict)
        load_amp = load_cfg is None or load_cfg.get("amp", True)
        if load_amp and "amp_state" in loaded_dict:
            load_optimizer = load_cfg is None or bool(load_cfg.get("optimizer"))
            self.amp.load_checkpoint_state(loaded_dict["amp_state"], load_optimizer, strict)
        return load_iteration
