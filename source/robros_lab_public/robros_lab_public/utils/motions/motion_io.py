"""Canonical DOF-based motion pickle I/O.

Canonical motion files contain one direct dictionary per clip.  Body states are
intentionally not persisted: the configured MJCF is the source of truth for
forward kinematics at runtime.
"""

from __future__ import annotations

import os
import pickle
import tempfile
import warnings
from pathlib import Path
from typing import Any

import numpy as np

# These fields define the required loading contract.  Motion PKLs may also
# contain auxiliary metadata; the loader ignores any additional top-level
# fields.
CANONICAL_KEYS = frozenset({"fps", "root_pos", "root_rot", "root_vel", "root_angvel", "dof_pos", "dof_vel"})
TIME_SERIES_KEYS = ("root_pos", "root_rot", "root_vel", "root_angvel", "dof_pos", "dof_vel")


def _as_float32(value: Any, key: str) -> np.ndarray:
    array = np.ascontiguousarray(np.asarray(value, dtype=np.float32))
    if not np.isfinite(array).all():
        raise ValueError(f"Motion field '{key}' contains NaN or infinity.")
    return array


def normalize_quaternion(quat: np.ndarray) -> np.ndarray:
    quat = _as_float32(quat, "root_rot")
    norm = np.linalg.norm(quat, axis=-1, keepdims=True)
    if np.any(norm < 1.0e-8):
        raise ValueError("Motion field 'root_rot' contains a zero-length quaternion.")
    quat = quat / norm
    if quat.shape[0] > 1:
        signs = np.ones((quat.shape[0], 1), dtype=np.float32)
        signs[1:, 0] = np.where(np.sum(quat[1:] * quat[:-1], axis=-1) < 0.0, -1.0, 1.0)
        signs = np.cumprod(signs, axis=0)
        quat = quat * signs
    return np.ascontiguousarray(quat, dtype=np.float32)


def validate_motion_clip(payload: Any, expected_num_dof: int | None = None) -> dict[str, Any]:
    if not isinstance(payload, dict):
        raise TypeError(f"Canonical motion payload must be a direct dictionary, got {type(payload).__name__}.")
    missing = sorted(CANONICAL_KEYS - set(payload))
    if missing:
        raise KeyError(f"Canonical motion payload is missing required keys: {missing}.")
    fps_values = np.asarray(payload["fps"], dtype=np.float32).reshape(-1)
    if fps_values.size != 1 or not np.isfinite(fps_values[0]) or fps_values[0] <= 0.0:
        raise ValueError(f"Motion field 'fps' must be one finite positive scalar, got {payload['fps']!r}.")
    arrays = {key: _as_float32(payload[key], key) for key in TIME_SERIES_KEYS}
    if arrays["root_pos"].ndim != 2 or arrays["root_pos"].shape[1] != 3:
        raise ValueError(f"Motion field 'root_pos' must have shape [T, 3], got {arrays['root_pos'].shape}.")
    if arrays["root_rot"].ndim != 2 or arrays["root_rot"].shape[1] != 4:
        raise ValueError(f"Motion field 'root_rot' must have shape [T, 4], got {arrays['root_rot'].shape}.")
    for key in ("root_vel", "root_angvel"):
        if arrays[key].shape != arrays["root_pos"].shape:
            raise ValueError(f"Motion field '{key}' shape {arrays[key].shape} does not match root_pos.")
    for key in ("dof_pos", "dof_vel"):
        if arrays[key].ndim != 2 or arrays[key].shape != arrays["dof_pos"].shape:
            raise ValueError(f"Motion field '{key}' must have the same [T, J] shape as dof_pos.")
    frame_count = arrays["root_pos"].shape[0]
    if frame_count < 1 or any(array.shape[0] != frame_count for array in arrays.values()):
        raise ValueError("All canonical motion fields must have the same non-zero frame count.")
    if expected_num_dof is not None and arrays["dof_pos"].shape[1] != expected_num_dof:
        raise ValueError(
            f"Canonical motion has {arrays['dof_pos'].shape[1]} DoFs but the configured MJCF exposes "
            f"{expected_num_dof}."
        )
    arrays["root_rot"] = normalize_quaternion(arrays["root_rot"])
    return {"fps": float(fps_values[0]), **arrays}


def load_motion_clip(path: str | Path, expected_num_dof: int | None = None) -> dict[str, Any]:
    motion_path = Path(path).expanduser().resolve()
    if motion_path.suffix.lower() != ".pkl":
        raise ValueError(f"Canonical motion loader accepts .pkl only, got '{motion_path}'.")
    if not motion_path.is_file():
        raise FileNotFoundError(f"Canonical motion file does not exist: {motion_path}")
    with motion_path.open("rb") as file:
        payload = pickle.load(file)
    return validate_motion_clip(payload, expected_num_dof=expected_num_dof)


def save_motion_clip(path: str | Path, payload: dict[str, Any], expected_num_dof: int | None = None) -> None:
    motion_path = Path(path).expanduser().resolve()
    motion_path.parent.mkdir(parents=True, exist_ok=True)
    validated = validate_motion_clip(payload, expected_num_dof=expected_num_dof)
    fd, temporary_name = tempfile.mkstemp(prefix=f".{motion_path.name}.", suffix=".tmp", dir=motion_path.parent)
    try:
        with os.fdopen(fd, "wb") as file:
            pickle.dump(validated, file, protocol=pickle.HIGHEST_PROTOCOL)
            file.flush()
            os.fsync(file.fileno())
        os.replace(temporary_name, motion_path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def discover_motion_files(motion_dir: str | Path) -> list[Path]:
    root = Path(motion_dir).expanduser().resolve()
    if not root.is_dir():
        raise NotADirectoryError(f"Motion loader expects a directory of .pkl files, got: {root}")
    metadata_files = [path for path in root.rglob("metadata.pkl") if path.is_file()]
    if metadata_files:
        warnings.warn(
            f"Ignoring legacy metadata.pkl files while discovering canonical motions: {metadata_files}",
            RuntimeWarning,
            stacklevel=2,
        )
    return sorted(
        (path for path in root.rglob("*.pkl") if path.is_file() and path.name != "metadata.pkl"),
        key=lambda path: path.relative_to(root).as_posix(),
    )


def finite_difference(values: np.ndarray, fps: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    result = np.zeros_like(values)
    if values.shape[0] <= 1:
        return result
    result[0] = (values[1] - values[0]) * fps
    result[-1] = (values[-1] - values[-2]) * fps
    if values.shape[0] > 2:
        result[1:-1] = (values[2:] - values[:-2]) * (0.5 * fps)
    return result


def resample_indices(
    frame_count: int, input_fps: float, output_fps: float
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if frame_count <= 1:
        zeros = np.zeros(1, dtype=np.int64)
        return zeros, zeros, np.zeros(1, dtype=np.float32)
    duration = (frame_count - 1) / input_fps
    times = np.arange(0.0, duration + 1.0e-8, 1.0 / output_fps, dtype=np.float64)
    positions = np.clip(times * input_fps, 0.0, frame_count - 1)
    index_0 = np.floor(positions).astype(np.int64)
    index_1 = np.minimum(index_0 + 1, frame_count - 1)
    return index_0, index_1, (positions - index_0).astype(np.float32)


def resample_linear(values: np.ndarray, index_0: np.ndarray, index_1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    blend = blend.reshape((blend.shape[0],) + (1,) * (values.ndim - 1))
    return np.ascontiguousarray(values[index_0] * (1.0 - blend) + values[index_1] * blend, dtype=np.float32)


def resample_quaternion(values: np.ndarray, index_0: np.ndarray, index_1: np.ndarray, blend: np.ndarray) -> np.ndarray:
    q0 = normalize_quaternion(values[index_0])
    q1 = normalize_quaternion(values[index_1])
    q1 = np.where(np.sum(q0 * q1, axis=-1, keepdims=True) < 0.0, -q1, q1)
    blend = blend.reshape((-1, 1))
    return normalize_quaternion(q0 * (1.0 - blend) + q1 * blend)
