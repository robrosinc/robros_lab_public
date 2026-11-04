# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Adapted from NVlabs/GR00T-WholeBodyControl (gear_sonic/isaac_utils/rotations.py).
# Source: https://github.com/NVlabs/GR00T-WholeBodyControl
# Modified by ROBROS.

"""Quaternion and transform helpers used by the copied Gear Sonic motion library."""

from __future__ import annotations

import math
from typing import Sequence

import torch
import torch.nn.functional as F


def normalize(x: torch.Tensor, eps: float = 1.0e-9) -> torch.Tensor:
    return x / torch.clamp(torch.linalg.norm(x, dim=-1, keepdim=True), min=eps)


def quat_unit(quat: torch.Tensor) -> torch.Tensor:
    return normalize(quat)


def quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """Return the conjugate of an ``xyzw`` quaternion."""
    return torch.cat((-quat[..., :3], quat[..., 3:4]), dim=-1)


def quat_inverse(quat: torch.Tensor) -> torch.Tensor:
    """Return the inverse of an ``xyzw`` unit quaternion."""
    return quat_conjugate(normalize(quat))


def quat_mul(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    """Multiply two ``xyzw`` quaternions."""
    quat_a, quat_b = torch.broadcast_tensors(quat_a, quat_b)
    x1, y1, z1, w1 = torch.unbind(quat_a, dim=-1)
    x2, y2, z2, w2 = torch.unbind(quat_b, dim=-1)

    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2
    return torch.stack((x, y, z, w), dim=-1)


def quat_mul_norm(quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
    return normalize(quat_mul(quat_a, quat_b))


def quat_rotate(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by ``xyzw`` quaternions."""
    quat, vec = torch.broadcast_tensors(quat, torch.cat((vec, vec[..., :1]), dim=-1))
    vec = vec[..., :3]
    xyz, w = quat[..., :3], quat[..., 3:4]
    t = 2.0 * torch.cross(xyz, vec, dim=-1)
    return vec + w * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    return quat_rotate(quat_inverse(quat), vec)


def quat_angle_axis(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Convert an ``xyzw`` quaternion to angle-axis form."""
    quat = normalize(quat)
    xyz, w = quat[..., :3], quat[..., 3:4]
    xyz_norm = torch.linalg.norm(xyz, dim=-1, keepdim=True)
    angle = 2.0 * torch.atan2(xyz_norm, torch.clamp(w, min=-1.0, max=1.0))
    angle = torch.where(angle > math.pi, angle - 2.0 * math.pi, angle)
    default_axis = torch.zeros_like(xyz)
    default_axis[..., 2] = 1.0
    axis = torch.where(xyz_norm > 1.0e-8, xyz / torch.clamp(xyz_norm, min=1.0e-8), default_axis)
    return angle.squeeze(-1), axis


def quat_to_angle_axis(quat: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    return quat_angle_axis(quat)


def quat_to_exp_map(quat: torch.Tensor) -> torch.Tensor:
    angle, axis = quat_to_angle_axis(quat)
    return angle.unsqueeze(-1) * axis


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    axis = normalize(axis)
    half_angle = 0.5 * angle.unsqueeze(-1)
    xyz = axis * torch.sin(half_angle)
    w = torch.cos(half_angle)
    return normalize(torch.cat((xyz, w), dim=-1))


def slerp(quat_0: torch.Tensor, quat_1: torch.Tensor, blend: torch.Tensor) -> torch.Tensor:
    quat_0 = normalize(quat_0)
    quat_1 = normalize(quat_1)
    while blend.ndim < quat_0.ndim:
        blend = blend.unsqueeze(-1)

    dot = torch.sum(quat_0 * quat_1, dim=-1, keepdim=True)
    quat_1 = torch.where(dot < 0.0, -quat_1, quat_1)
    dot = torch.clamp(torch.abs(dot), max=1.0)

    theta_0 = torch.acos(dot)
    sin_theta_0 = torch.sin(theta_0)
    theta = theta_0 * blend
    sin_theta = torch.sin(theta)
    coeff_0 = torch.cos(theta) - dot * sin_theta / torch.clamp(sin_theta_0, min=1.0e-8)
    coeff_1 = sin_theta / torch.clamp(sin_theta_0, min=1.0e-8)
    linear = normalize((1.0 - blend) * quat_0 + blend * quat_1)
    spherical = coeff_0 * quat_0 + coeff_1 * quat_1
    return normalize(torch.where(sin_theta_0 < 1.0e-6, linear, spherical))


def quat_identity(
    shape: Sequence[int],
    *,
    device: torch.device | str | None = None,
    dtype: torch.dtype | None = None,
) -> torch.Tensor:
    quat = torch.zeros(*shape, 4, device=device, dtype=dtype or torch.float32)
    quat[..., 3] = 1.0
    return quat


def quat_identity_like(quat: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(quat)
    out[..., 3] = 1.0
    return out


def transform_from_rotation_translation(
    r: torch.Tensor | None = None,
    t: torch.Tensor | None = None,
) -> torch.Tensor:
    if r is None and t is None:
        raise ValueError("rotation and translation cannot both be None")
    if r is None:
        assert t is not None
        r = quat_identity(t.shape[:-1], device=t.device, dtype=t.dtype)
    if t is None:
        t = torch.zeros(*r.shape[:-1], 3, device=r.device, dtype=r.dtype)
    return torch.cat((r, t), dim=-1)


def transform_rotation(transform: torch.Tensor) -> torch.Tensor:
    return transform[..., :4]


def transform_translation(transform: torch.Tensor) -> torch.Tensor:
    return transform[..., 4:]


def transform_mul(transform_a: torch.Tensor, transform_b: torch.Tensor) -> torch.Tensor:
    rot_a = transform_rotation(transform_a)
    rot_b = transform_rotation(transform_b)
    trans_a = transform_translation(transform_a)
    trans_b = transform_translation(transform_b)
    return transform_from_rotation_translation(
        r=quat_mul_norm(rot_a, rot_b),
        t=quat_rotate(rot_a, trans_b) + trans_a,
    )


def transform_inverse(transform: torch.Tensor) -> torch.Tensor:
    rot_inv = quat_inverse(transform_rotation(transform))
    trans_inv = -quat_rotate(rot_inv, transform_translation(transform))
    return transform_from_rotation_translation(r=rot_inv, t=trans_inv)


def quaternion_to_matrix(quaternions: torch.Tensor) -> torch.Tensor:
    quaternions = normalize(quaternions)
    i, j, k, r = torch.unbind(quaternions, dim=-1)
    two_s = 2.0 / torch.clamp((quaternions * quaternions).sum(dim=-1), min=1.0e-8)
    matrix = torch.stack(
        (
            1.0 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1.0 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1.0 - two_s * (i * i + j * j),
        ),
        dim=-1,
    )
    return matrix.reshape(quaternions.shape[:-1] + (3, 3))


def axis_angle_to_quaternion(axis_angle: torch.Tensor) -> torch.Tensor:
    angle = torch.linalg.norm(axis_angle, dim=-1, keepdim=True)
    half_angle = 0.5 * angle
    scale = torch.where(
        angle > 1.0e-8,
        torch.sin(half_angle) / torch.clamp(angle, min=1.0e-8),
        0.5 - angle * angle / 48.0,
    )
    return normalize(torch.cat((axis_angle * scale, torch.cos(half_angle)), dim=-1))


def _sqrt_positive_part(x: torch.Tensor) -> torch.Tensor:
    out = torch.zeros_like(x)
    mask = x > 0.0
    out[mask] = torch.sqrt(x[mask])
    return out


def matrix_to_quaternion(matrix: torch.Tensor) -> torch.Tensor:
    if matrix.size(-1) != 3 or matrix.size(-2) != 3:
        raise ValueError(f"Invalid rotation matrix shape {matrix.shape}.")
    batch_dim = matrix.shape[:-2]
    m00, m01, m02, m10, m11, m12, m20, m21, m22 = torch.unbind(matrix.reshape(batch_dim + (9,)), dim=-1)
    q_abs = _sqrt_positive_part(
        torch.stack(
            (
                1.0 + m00 + m11 + m22,
                1.0 + m00 - m11 - m22,
                1.0 - m00 + m11 - m22,
                1.0 - m00 - m11 + m22,
            ),
            dim=-1,
        )
    )
    quat_by_rijk = torch.stack(
        (
            torch.stack((q_abs[..., 0] ** 2, m21 - m12, m02 - m20, m10 - m01), dim=-1),
            torch.stack((m21 - m12, q_abs[..., 1] ** 2, m10 + m01, m02 + m20), dim=-1),
            torch.stack((m02 - m20, m10 + m01, q_abs[..., 2] ** 2, m12 + m21), dim=-1),
            torch.stack((m10 - m01, m20 + m02, m21 + m12, q_abs[..., 3] ** 2), dim=-1),
        ),
        dim=-2,
    )
    floor = torch.tensor(0.1, dtype=q_abs.dtype, device=q_abs.device)
    quat_candidates = quat_by_rijk / (2.0 * q_abs[..., None].max(floor))
    selected = F.one_hot(q_abs.argmax(dim=-1), num_classes=4) > 0.5
    quat_wxyz = normalize(quat_candidates[selected, :].reshape(batch_dim + (4,)))
    return wxyz_to_xyzw(quat_wxyz)


def wxyz_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    return quat[..., [1, 2, 3, 0]]


def quat_from_euler_xyz(roll: torch.Tensor, pitch: torch.Tensor, yaw: torch.Tensor) -> torch.Tensor:
    cy = torch.cos(yaw * 0.5)
    sy = torch.sin(yaw * 0.5)
    cr = torch.cos(roll * 0.5)
    sr = torch.sin(roll * 0.5)
    cp = torch.cos(pitch * 0.5)
    sp = torch.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return torch.stack((qx, qy, qz, qw), dim=-1)
