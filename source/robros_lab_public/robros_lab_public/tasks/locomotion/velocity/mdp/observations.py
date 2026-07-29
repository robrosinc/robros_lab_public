from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.managers import ManagerTermBase, ObservationTermCfg


class gait_phase_obs(ManagerTermBase):
    """Sinusoidal gait phase observation driven by the ``gait_action`` term."""

    def __init__(self, cfg: ObservationTermCfg, env: ManagerBasedRLEnv):
        super().__init__(cfg, env)
        self.dt = env.step_dt
        self.global_gait_phase = torch.zeros(env.num_envs, device=env.device)
        self._default_gait_freq = cfg.params.get("default_gait_freq", 1.0)

    def __call__(self, env: ManagerBasedRLEnv, asset_cfg, default_gait_freq=1.0) -> torch.Tensor:
        del asset_cfg, default_gait_freq

        gait_freq = env.action_manager.get_term("gait_action").processed_actions.squeeze(-1)  # type: ignore[attr-defined]
        self.global_gait_phase = torch.frac(self.global_gait_phase + self.dt * gait_freq * self._default_gait_freq)
        velcmd = env.command_manager.get_command("base_velocity")
        slow = torch.norm(velcmd[:, :3], dim=-1) < 0.2
        phase = torch.cat(
            (self.global_gait_phase.unsqueeze(dim=-1), self.global_gait_phase.unsqueeze(dim=-1) + 0.5), dim=-1
        )
        self.global_gait_phase[slow] = 0.25
        phase[slow] = 0.25
        return torch.sin(2 * torch.pi * phase)


def gait_action(env: ManagerBasedRLEnv) -> torch.Tensor:
    """Return the processed gait-frequency action."""
    return env.action_manager.get_term("gait_action").processed_actions


__all__ = ["gait_action", "gait_phase_obs"]
