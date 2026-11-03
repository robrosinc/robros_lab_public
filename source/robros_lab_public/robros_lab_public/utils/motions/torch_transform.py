# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVlabs/GR00T-WholeBodyControl (gear_sonic/trl/utils/torch_transform.py).
# Source: https://github.com/NVlabs/GR00T-WholeBodyControl
# Modified by ROBROS.

"""Small quaternion conversion subset used by the Gear Sonic motion copy."""

from __future__ import annotations

import torch

from . import rotations


def angle_axis_to_quaternion(angle_axis: torch.Tensor) -> torch.Tensor:
    """Convert angle-axis vectors to scalar-last ``xyzw`` quaternions."""
    return rotations.axis_angle_to_quaternion(angle_axis)


def quaternion_to_angle_axis(quaternion: torch.Tensor) -> torch.Tensor:
    """Convert scalar-last ``xyzw`` quaternions to angle-axis vectors."""
    angle, axis = rotations.quat_angle_axis(quaternion)
    return axis * angle.unsqueeze(-1)


def quaternion_to_rotation_matrix(quaternion: torch.Tensor) -> torch.Tensor:
    return rotations.quaternion_to_matrix(quaternion)


def rotation_matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    return rotations.matrix_to_quaternion(matrix)


def angle_axis_to_rotation_matrix(angle_axis: torch.Tensor) -> torch.Tensor:
    return rotations.quaternion_to_matrix(angle_axis_to_quaternion(angle_axis))


def rotation_matrix_to_angle_axis(matrix: torch.Tensor) -> torch.Tensor:
    return quaternion_to_angle_axis(rotation_matrix_to_quaternion(matrix))
