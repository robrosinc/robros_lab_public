# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Custom ManagerBasedRLEnv with IGRIS-C specific buffers (phase, randomization tracking)."""

from __future__ import annotations

import torch
from isaaclab.envs import ManagerBasedRLEnv


class RobrosVelocityRLEnv(ManagerBasedRLEnv):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._validate_amp_group()

    def step(self, action: torch.Tensor):
        """Execute one step and store terminal AMP observations inside ``extras['amp']``."""
        self.action_manager.process_action(action.to(self.device))
        self.recorder_manager.record_pre_step()

        is_rendering = self.sim.is_rendering

        if self._physics_handles_decimation:
            self._sim_step_counter += self.cfg.decimation
            self.action_manager.apply_action()
            self.scene.write_data_to_sim()
            self.sim.step(render=False)
            self.recorder_manager.record_post_physics_decimation_step()
            if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                self.sim.render(skip_app_pumping=not self.render_enabled)
            self.scene.update(dt=self.step_dt)
        else:
            for _ in range(self.cfg.decimation):
                self._sim_step_counter += 1
                self.action_manager.apply_action()
                self.scene.write_data_to_sim()
                self.sim.step(render=False)
                self.recorder_manager.record_post_physics_decimation_step()
                if self._sim_step_counter % self.cfg.sim.render_interval == 0 and is_rendering:
                    self.sim.render(skip_app_pumping=not self.render_enabled)
                self.scene.update(dt=self.physics_dt)

        self.episode_length_buf += 1
        self.common_step_counter += 1
        self.reset_buf = self.termination_manager.compute()
        self.reset_terminated = self.termination_manager.terminated
        self.reset_time_outs = self.termination_manager.time_outs
        self.reward_buf = self.reward_manager.compute(dt=self.step_dt)

        if len(self.recorder_manager.active_terms) > 0:
            self.obs_buf = self.observation_manager.compute()
            self.recorder_manager.record_post_step()

        self.extras["amp"] = {}
        reset_env_ids = self.reset_buf.nonzero(as_tuple=False).squeeze(-1).int()
        if len(reset_env_ids) > 0:
            terminal_amp_obs = self.observation_manager.compute_group("amp", update_history=False).clone()
            self.extras["amp"]["terminal_observation"] = terminal_amp_obs

            self.recorder_manager.record_pre_reset(reset_env_ids)
            self._reset_idx(reset_env_ids)

            if self.render_enabled and is_rendering and self.has_rtx_sensors and self.cfg.num_rerenders_on_reset > 0:
                for _ in range(self.cfg.num_rerenders_on_reset):
                    self.sim.render()

            self.recorder_manager.record_post_reset(reset_env_ids)

        self.command_manager.compute(dt=self.step_dt)
        if "interval" in self.event_manager.available_modes:
            self.event_manager.apply(mode="interval", dt=self.step_dt)
        self.obs_buf = self.observation_manager.compute(update_history=True)
        return self.obs_buf, self.reward_buf, self.reset_terminated, self.reset_time_outs, self.extras

    def _validate_amp_group(self) -> None:
        active_terms = getattr(self.observation_manager, "active_terms", {})
        if "amp" not in active_terms:
            raise ValueError("RobrosVelocityRLEnv requires an 'amp' observation group in the environment config.")

        group_term_cfgs = getattr(self.observation_manager, "_group_obs_term_cfgs", {}).get("amp", [])
        for term_name, term_cfg in zip(active_terms["amp"], group_term_cfgs):
            history_length = int(getattr(term_cfg, "history_length", 0))
            if history_length > 0:
                raise ValueError(
                    f"AMP observation term 'amp.{term_name}' uses history_length={history_length}."
                    " AMP observation groups must be history-free."
                )
