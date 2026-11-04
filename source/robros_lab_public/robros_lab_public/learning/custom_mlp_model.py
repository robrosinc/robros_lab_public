# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import copy
import torch
import torch.nn as nn
from tensordict import TensorDict

from rsl_rl.models import MLPModel
from rsl_rl.modules import EmpiricalNormalization, HiddenState


class CustomMLPModel(MLPModel):
    """MLP model with explicit per-observation-group normalization.

    The model consumes the observation groups declared in ``obs_groups[obs_set]``
    exactly like :class:`MLPModel`. ``normalized_obs_groups`` selects which of
    those groups are empirically normalized before all active groups are
    concatenated and passed to the shared MLP.
    """

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        obs_set: str,
        output_dim: int,
        hidden_dims: tuple[int, ...] | list[int] = (256, 256, 256),
        activation: str = "elu",
        obs_normalization: bool = False,
        normalized_obs_groups: tuple[str, ...] | list[str] | None = None,
        distribution_cfg: dict | None = None,
    ) -> None:
        """Build an MLP and resolve which active observation groups to normalize."""
        super().__init__(
            obs,
            obs_groups,
            obs_set,
            output_dim,
            hidden_dims=hidden_dims,
            activation=activation,
            obs_normalization=False,
            distribution_cfg=distribution_cfg,
        )

        self.obs_group_dims = {group: int(obs[group].shape[-1]) for group in self.obs_groups}
        self.obs_group_slices = self._build_obs_group_slices()
        self.normalized_obs_groups = self._resolve_normalized_obs_groups(normalized_obs_groups, obs_normalization)
        self.normalization_obs_groups = self.normalized_obs_groups
        self._normalization_indices = self._build_normalization_indices(self.normalized_obs_groups)
        self.obs_normalization = bool(self._normalization_indices.numel() > 0)
        if self.obs_normalization:
            self.obs_normalizer = EmpiricalNormalization(int(self._normalization_indices.numel()))
        else:
            self.obs_normalizer = nn.Identity()

    def get_latent(
        self, obs: TensorDict, masks: torch.Tensor | None = None, hidden_state: HiddenState = None
    ) -> torch.Tensor:
        """Concatenate all active groups and normalize only the configured subset."""
        del masks, hidden_state
        latent = torch.cat([obs[obs_group] for obs_group in self.obs_groups], dim=-1)
        return self._normalize_latent(latent)

    def update_normalization(self, obs: TensorDict) -> None:
        """Update statistics from only the configured observation groups."""
        if not self.obs_normalization:
            return
        normalized_obs = torch.cat([obs[group] for group in self.normalized_obs_groups], dim=-1)
        self.obs_normalizer.update(normalized_obs)  # type: ignore[arg-type]

    def as_jit(self) -> nn.Module:
        """Return a version of the model compatible with Torch JIT export."""
        return _TorchCustomMLPModel(self)

    def as_onnx(self, verbose: bool) -> nn.Module:
        """Return a version of the model compatible with ONNX export."""
        return _OnnxCustomMLPModel(self, verbose)

    def _normalize_latent(self, latent: torch.Tensor) -> torch.Tensor:
        if not self.obs_normalization:
            return latent
        indices = self._normalization_indices.to(device=latent.device)
        normalized_values = self.obs_normalizer(latent.index_select(-1, indices))
        return latent.scatter(-1, indices.expand_as(normalized_values), normalized_values)

    def _build_obs_group_slices(self) -> dict[str, slice]:
        slices = {}
        start = 0
        for obs_group in self.obs_groups:
            end = start + self.obs_group_dims[obs_group]
            slices[obs_group] = slice(start, end)
            start = end
        return slices

    def _resolve_normalized_obs_groups(
        self,
        normalized_obs_groups: tuple[str, ...] | list[str] | None,
        obs_normalization: bool,
    ) -> tuple[str, ...]:
        if normalized_obs_groups is None:
            return tuple(self.obs_groups) if obs_normalization else ()

        requested = tuple(str(group) for group in normalized_obs_groups)
        duplicates = sorted({group for group in requested if requested.count(group) > 1})
        if duplicates:
            raise ValueError(f"CustomMLPModel normalized_obs_groups contains duplicates: {duplicates}.")

        active_groups = set(self.obs_groups)
        unknown_groups = [group for group in requested if group not in active_groups]
        if unknown_groups:
            raise ValueError(
                "CustomMLPModel normalized_obs_groups contains groups that are not active for this model: "
                f"{unknown_groups}. Active groups: {list(self.obs_groups)}."
            )
        requested_groups = set(requested)
        return tuple(group for group in self.obs_groups if group in requested_groups)

    def _build_normalization_indices(self, normalized_obs_groups: tuple[str, ...]) -> torch.Tensor:
        indices = []
        for group in normalized_obs_groups:
            group_slice = self.obs_group_slices[group]
            indices.extend(range(group_slice.start, group_slice.stop))
        return torch.tensor(indices, dtype=torch.long)


class _TorchCustomMLPModel(nn.Module):
    """TorchScript-compatible deterministic CustomMLPModel export."""

    def __init__(self, model: CustomMLPModel) -> None:
        super().__init__()
        self.obs_normalization = model.obs_normalization
        self.obs_normalizer = copy.deepcopy(model.obs_normalizer)
        self.register_buffer("normalization_indices", model._normalization_indices.clone())
        self.mlp = copy.deepcopy(model.mlp)
        if model.distribution is not None:
            self.deterministic_output = model.distribution.as_deterministic_output_module()
        else:
            self.deterministic_output = nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self._normalize_input(x)
        out = self.mlp(x)
        return self.deterministic_output(out)

    def _normalize_input(self, x: torch.Tensor) -> torch.Tensor:
        if not self.obs_normalization:
            return x
        normalized_values = self.obs_normalizer(x.index_select(-1, self.normalization_indices))
        return x.scatter(-1, self.normalization_indices.expand_as(normalized_values), normalized_values)

    @torch.jit.export
    def reset(self) -> None:
        pass


class _OnnxCustomMLPModel(_TorchCustomMLPModel):
    """ONNX-compatible deterministic CustomMLPModel export."""

    is_recurrent: bool = False

    def __init__(self, model: CustomMLPModel, verbose: bool) -> None:
        super().__init__(model)
        self.verbose = verbose
        self.input_size = model.obs_dim

    def get_dummy_inputs(self) -> tuple[torch.Tensor]:
        return (torch.zeros(1, self.input_size),)

    @property
    def input_names(self) -> list[str]:
        return ["obs"]

    @property
    def output_names(self) -> list[str]:
        return ["actions"]
