from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedEnv
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass


class GaitFreqAction(ActionTerm):
    """Scalar gait-frequency action that only updates environment buffers."""

    def __init__(self, cfg: GaitFreqActionCfg, env: ManagerBasedEnv):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((self.num_envs, 1), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._lpf_alpha = getattr(cfg, "lpf_alpha", 0.3)

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor):
        self._raw_actions.copy_(actions)
        scaled_actions = torch.tanh(self._raw_actions) * 0.4 + 1.0
        self._processed_actions = self._lpf_alpha * self._processed_actions + (1.0 - self._lpf_alpha) * scaled_actions

    def apply_actions(self):
        return

    def reset(self, env_ids=None):
        if env_ids is None:
            env_ids = slice(None)
        self._raw_actions[env_ids] = 1.0
        self._processed_actions[env_ids] = 1.0


@configclass
class GaitFreqActionCfg(ActionTermCfg):
    class_type: type[ActionTerm] = GaitFreqAction
    lpf_alpha: float = 0.3


__all__ = ["GaitFreqAction", "GaitFreqActionCfg"]
