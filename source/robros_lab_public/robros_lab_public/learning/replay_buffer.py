# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import torch
from collections.abc import Generator


class ReplayBuffer:
    """Fixed-size circular replay buffer for AMP transition windows."""

    def __init__(self, obs_dim: int, buffer_size: int, device: str = "cpu") -> None:
        """Allocate storage for flattened transition windows."""
        self.transitions = torch.zeros(buffer_size, obs_dim, device=device)
        self.buffer_size = buffer_size
        self.device = device
        self.step = 0
        self.num_samples = 0

    def insert(self, transitions: torch.Tensor) -> None:
        """Append a batch of flattened transitions, overwriting the oldest data when full."""
        if transitions.ndim != 2:
            raise ValueError(f"ReplayBuffer expects 2D transition batches, got shape {transitions.shape}.")
        if transitions.shape[1] != self.transitions.shape[1]:
            raise ValueError(
                f"ReplayBuffer transition width mismatch: expected {self.transitions.shape[1]}, got {transitions.shape[1]}."
            )

        batch_size = transitions.shape[0]
        start = self.step
        end = start + batch_size

        if end > self.buffer_size:
            first = self.buffer_size - start
            second = end - self.buffer_size
            self.transitions[start:] = transitions[:first]
            self.transitions[:second] = transitions[first:]
        else:
            self.transitions[start:end] = transitions

        self.step = end % self.buffer_size
        self.num_samples = min(self.buffer_size, max(self.num_samples, end))

    def feed_forward_generator(
        self, num_mini_batches: int, mini_batch_size: int
    ) -> Generator[torch.Tensor, None, None]:
        """Yield uniformly sampled mini-batches of transitions."""
        if self.num_samples == 0:
            raise RuntimeError("ReplayBuffer is empty. Insert transitions before requesting batches.")

        for _ in range(num_mini_batches):
            indices = torch.randint(0, self.num_samples, (mini_batch_size,), device=self.device)
            yield self.transitions[indices]
