"""Directory-based canonical DOF motion loader."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

from robros_lab_public.utils.motions import rotations
from robros_lab_public.utils.motions.motion_io import (
    discover_motion_files,
    finite_difference,
    load_motion_clip,
    resample_indices,
    resample_linear,
    resample_quaternion,
)
from robros_lab_public.utils.motions.torch_humanoid_batch import Humanoid_Batch


class _FKConfig(dict):
    def __init__(self, mjcf_path: Path) -> None:
        super().__init__(load_mesh=False)
        self.asset = SimpleNamespace(assetRoot=str(mjcf_path.parent), assetFileName=mjcf_path.name)
        self.extend_config = []


class MotionLoader:
    """Load canonical DOF PKL clips and derive body states with MJCF FK."""

    def __init__(
        self,
        motion_dir: str,
        body_names: Sequence[str],
        joint_names: Sequence[str],
        mjcf_file: str | Path,
        local_frame_body_name: str = "base_link",
        device: str | torch.device = "cpu",
        storage_device: str | torch.device | None = None,
        max_resident_motions: int | None = None,
        target_fps: int = 50,
        num_envs: int = 1,
        freeze_frame_aug: bool = False,
        freeze_frame_prob: float = 0.1,
        freeze_frame_seed: int | None = None,
    ) -> None:
        del num_envs
        self.device = torch.device(device)
        self.storage_device = torch.device(storage_device) if storage_device is not None else self.device
        self.motion_dir = str(Path(motion_dir).expanduser().resolve())
        self.local_frame_body_name = str(local_frame_body_name)
        self._requested_body_names = tuple(str(name) for name in body_names)
        self._requested_joint_names = tuple(str(name) for name in joint_names)
        self.target_fps = float(target_fps)
        self.freeze_frame_aug = bool(freeze_frame_aug)
        self.freeze_frame_prob = float(freeze_frame_prob)
        if not 0.0 <= self.freeze_frame_prob <= 1.0:
            raise ValueError(f"freeze_frame_prob must be in [0, 1], got {self.freeze_frame_prob}.")
        # Match SONIC's use of NumPy's process-level RNG by default so the
        # experiment seed controls augmentation.  A private generator remains
        # available for deterministic loader-level tests and data inspection.
        self._freeze_frame_rng = None if freeze_frame_seed is None else np.random.default_rng(freeze_frame_seed)
        self.max_resident_motions = None if max_resident_motions is None else int(max_resident_motions)
        self._motion_paths = discover_motion_files(self.motion_dir)
        if not self._motion_paths:
            raise FileNotFoundError(f"Motion directory contains no canonical .pkl clips: {self.motion_dir}")
        mjcf_path = Path(mjcf_file).expanduser().resolve()
        if not mjcf_path.is_file():
            raise FileNotFoundError(f"Configured MJCF file does not exist: {mjcf_path}")
        fk_cfg = _FKConfig(mjcf_path)
        self.fk = Humanoid_Batch(fk_cfg, device=self.storage_device)
        self._available_body_names = tuple(str(name) for name in self.fk.body_names)
        self._available_joint_names = tuple(str(name) for name in self.fk.joint_names)
        if len(self._available_joint_names) != self.fk.num_dof:
            raise ValueError("MJCF FK reported inconsistent canonical DoF joint names.")
        missing_bodies = [name for name in self._requested_body_names if name not in self._available_body_names]
        missing_joints = [name for name in self._requested_joint_names if name not in self._available_joint_names]
        if missing_bodies:
            raise ValueError(f"Configured MJCF is missing requested bodies: {missing_bodies}")
        if missing_joints:
            raise ValueError(f"Configured MJCF is missing requested joints: {missing_joints}")
        if self.local_frame_body_name not in self._requested_body_names:
            raise ValueError(f"local_frame_body_name='{self.local_frame_body_name}' must be in body_names.")
        self.body_names = self._requested_body_names
        self.joint_names = self._requested_joint_names
        self.local_frame_body_index = self.body_names.index(self.local_frame_body_name)
        self._body_indices = torch.as_tensor(
            [self._available_body_names.index(name) for name in self.body_names],
            dtype=torch.long,
            device=self.storage_device,
        )
        self._joint_indices = torch.as_tensor(
            [self._available_joint_names.index(name) for name in self.joint_names],
            dtype=torch.long,
            device=self.storage_device,
        )
        if self.max_resident_motions is None:
            self.max_resident_motions = len(self._motion_paths)
        if self.max_resident_motions <= 0:
            raise ValueError(f"max_resident_motions must be positive, got {self.max_resident_motions}.")
        self._resident_cursor = 0
        self._resident_order = np.random.default_rng(0).permutation(len(self._motion_paths)).tolist()
        self._load_resident_paths(self._next_resident_paths())

    def _next_resident_paths(self) -> list[Path]:
        count = min(self.max_resident_motions, len(self._motion_paths))
        if self._resident_cursor + count > len(self._resident_order):
            self._resident_order = (
                np.random.default_rng(self._resident_cursor + len(self._motion_paths))
                .permutation(len(self._motion_paths))
                .tolist()
            )
            self._resident_cursor = 0
        indexes = self._resident_order[self._resident_cursor : self._resident_cursor + count]
        self._resident_cursor += count
        return [self._motion_paths[index] for index in indexes]

    def _to_storage(self, array: np.ndarray) -> torch.Tensor:
        tensor = torch.as_tensor(array, dtype=torch.float32, device=self.storage_device)
        if self.storage_device.type == "cpu" and self.device.type == "cuda":
            tensor = tensor.pin_memory()
        return tensor

    def _resample_clip(self, payload: dict[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        input_fps = float(payload["fps"])
        frame_count = int(np.asarray(payload["dof_pos"]).shape[0])
        index_0, index_1, blend = resample_indices(frame_count, input_fps, self.target_fps)
        root_pos = resample_linear(np.asarray(payload["root_pos"]), index_0, index_1, blend)
        root_rot = resample_quaternion(np.asarray(payload["root_rot"]), index_0, index_1, blend)
        dof_pos = resample_linear(np.asarray(payload["dof_pos"]), index_0, index_1, blend)
        return self._to_storage(root_pos), self._to_storage(root_rot), self._to_storage(dof_pos)

    def _sample_freeze_frame_index(self, frame_count: int) -> int | None:
        """Sample SONIC-style freeze augmentation for one resampled clip."""
        if frame_count <= 0:
            raise ValueError(f"Cannot freeze a clip with {frame_count} frames.")
        if not self.freeze_frame_aug:
            return None
        random_value = (
            float(np.random.random()) if self._freeze_frame_rng is None else float(self._freeze_frame_rng.random())
        )
        if random_value >= self.freeze_frame_prob:
            return None
        if self._freeze_frame_rng is None:
            return int(np.random.randint(0, frame_count))
        return int(self._freeze_frame_rng.integers(0, frame_count))

    def _resample_and_augment_clip(
        self, payload: dict[str, Any]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int | None]:
        """Resample one clip and optionally hold a random pose through its tail."""
        root_pos, root_rot, dof_pos = self._resample_clip(payload)
        freeze_frame_index = self._sample_freeze_frame_index(int(dof_pos.shape[0]))
        if freeze_frame_index is not None:
            self._freeze_pose_streams((root_pos, root_rot, dof_pos), freeze_frame_index)
        return root_pos, root_rot, dof_pos, freeze_frame_index

    @staticmethod
    def _freeze_pose_streams(streams: Sequence[np.ndarray | torch.Tensor], freeze_frame_index: int) -> None:
        """Repeat one frame through the tail of aligned NumPy or Torch pose streams."""
        for stream in streams:
            frozen_frame = stream[freeze_frame_index : freeze_frame_index + 1]
            frozen_frame = frozen_frame.clone() if isinstance(frozen_frame, torch.Tensor) else frozen_frame.copy()
            stream[freeze_frame_index:] = frozen_frame

    def _build_local_streams(
        self,
        body_pos_w: torch.Tensor,
        body_quat_w: torch.Tensor,
        body_lin_vel_w: torch.Tensor,
        body_ang_vel_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_pos = body_pos_w[:, self.local_frame_body_index]
        anchor_quat = body_quat_w[:, self.local_frame_body_index]
        anchor_quat_expanded = anchor_quat[:, None, :].expand_as(body_quat_w)
        inverse = rotations.quat_inverse(anchor_quat_expanded)
        body_pos_b = rotations.quat_rotate(inverse, body_pos_w - anchor_pos[:, None, :])
        body_quat_b = rotations.quat_mul_norm(inverse, body_quat_w)
        body_lin_vel_b = rotations.quat_rotate(inverse, body_lin_vel_w)
        body_ang_vel_b = rotations.quat_rotate(inverse, body_ang_vel_w)
        return body_pos_b, body_quat_b, body_lin_vel_b, body_ang_vel_b

    def _load_resident_paths(self, paths: list[Path]) -> None:
        if not paths:
            raise ValueError("No resident motion clips were selected.")
        joint_pos_list: list[torch.Tensor] = []
        joint_vel_list: list[torch.Tensor] = []
        body_pos_list: list[torch.Tensor] = []
        body_quat_list: list[torch.Tensor] = []
        body_lin_vel_list: list[torch.Tensor] = []
        body_ang_vel_list: list[torch.Tensor] = []
        lengths: list[int] = []
        names: list[str] = []
        freeze_frame_indices: list[int | None] = []
        for path in paths:
            payload = load_motion_clip(path, expected_num_dof=self.fk.num_dof)
            root_pos, root_rot, dof_pos, freeze_frame_index = self._resample_and_augment_clip(payload)
            fk_result = self.fk.fk_from_dof(root_pos, root_rot, dof_pos, fps=self.target_fps, return_full=True)
            body_pos = fk_result.global_translation.index_select(1, self._body_indices)
            body_quat = fk_result.global_rotation.index_select(1, self._body_indices)
            body_lin_vel = fk_result.global_velocity.index_select(1, self._body_indices)
            body_ang_vel = fk_result.global_angular_velocity.index_select(1, self._body_indices)
            joint_pos = fk_result.dof_pos.index_select(1, self._joint_indices)
            joint_vel = fk_result.dof_vels.index_select(1, self._joint_indices)
            if freeze_frame_index is not None:
                # FK differentiators may retain a one-frame boundary velocity.
                # A held command must advertise an exact zero-velocity tail so
                # tracking rewards and encoders agree with the frozen poses.
                joint_vel[freeze_frame_index:] = 0.0
                body_lin_vel[freeze_frame_index:] = 0.0
                body_ang_vel[freeze_frame_index:] = 0.0
            body_pos_list.append(body_pos)
            body_quat_list.append(body_quat)
            body_lin_vel_list.append(body_lin_vel)
            body_ang_vel_list.append(body_ang_vel)
            joint_pos_list.append(joint_pos)
            joint_vel_list.append(joint_vel)
            lengths.append(int(joint_pos.shape[0]))
            names.append(path.relative_to(Path(self.motion_dir)).with_suffix("").as_posix())
            freeze_frame_indices.append(freeze_frame_index)

        self.fps = self.target_fps
        self.joint_pos = torch.cat(joint_pos_list, dim=0)
        self.joint_vel = torch.cat(joint_vel_list, dim=0)
        self.body_pos_w = torch.cat(body_pos_list, dim=0)
        self.body_quat_w = torch.cat(body_quat_list, dim=0)
        self.body_lin_vel_w = torch.cat(body_lin_vel_list, dim=0)
        self.body_ang_vel_w = torch.cat(body_ang_vel_list, dim=0)
        self.body_pos_b, self.body_quat_b, self.body_lin_vel_b, self.body_ang_vel_b = self._build_local_streams(
            self.body_pos_w, self.body_quat_w, self.body_lin_vel_w, self.body_ang_vel_w
        )
        self.body_lin_acc_w = self._clipwise_finite_difference(self.body_lin_vel_w, lengths)
        self.body_ang_acc_w = self._clipwise_finite_difference(self.body_ang_vel_w, lengths)
        self._zero_frozen_derivative_tails(self.body_lin_acc_w, lengths, freeze_frame_indices)
        self._zero_frozen_derivative_tails(self.body_ang_acc_w, lengths, freeze_frame_indices)
        if self.storage_device.type == "cpu" and self.device.type == "cuda":
            self.joint_pos = self.joint_pos.pin_memory()
            self.joint_vel = self.joint_vel.pin_memory()
            self.body_pos_w = self.body_pos_w.pin_memory()
            self.body_quat_w = self.body_quat_w.pin_memory()
            self.body_lin_vel_w = self.body_lin_vel_w.pin_memory()
            self.body_ang_vel_w = self.body_ang_vel_w.pin_memory()
            self.body_pos_b = self.body_pos_b.pin_memory()
            self.body_quat_b = self.body_quat_b.pin_memory()
            self.body_lin_vel_b = self.body_lin_vel_b.pin_memory()
            self.body_ang_vel_b = self.body_ang_vel_b.pin_memory()
            self.body_lin_acc_w = self.body_lin_acc_w.pin_memory()
            self.body_ang_acc_w = self.body_ang_acc_w.pin_memory()
        self.clip_lengths = torch.as_tensor(lengths, dtype=torch.long, device=self.device)
        self.clip_offsets = torch.cat(
            (torch.zeros(1, dtype=torch.long, device=self.device), torch.cumsum(self.clip_lengths, dim=0))
        )
        self.time_step_total = int(self.clip_offsets[-1].item())
        self.clip_count = len(lengths)
        self.max_clip_length = max(lengths)
        self.clip_paths = tuple(path.resolve() for path in paths)
        self.clip_names = tuple(names)
        self.freeze_frame_indices = tuple(freeze_frame_indices)
        self.clip_weights = torch.ones(self.clip_count, dtype=torch.float32, device=self.device)
        self.clip_durations_s = self.clip_lengths.to(dtype=torch.float32) / self.fps
        self.has_multiple_clips = self.clip_count > 1

    def _clipwise_finite_difference(self, values: torch.Tensor, lengths: list[int]) -> torch.Tensor:
        result = torch.zeros_like(values)
        start = 0
        for length in lengths:
            end = start + length
            if length > 1:
                result[start:end] = torch.as_tensor(
                    finite_difference(values[start:end].detach().cpu().numpy(), self.target_fps),
                    dtype=values.dtype,
                    device=values.device,
                )
            start = end
        return result

    @staticmethod
    def _zero_frozen_derivative_tails(
        values: torch.Tensor,
        lengths: Sequence[int],
        freeze_frame_indices: Sequence[int | None],
    ) -> None:
        """Zero derivative streams from each clip's frozen frame onward."""
        start = 0
        for length, freeze_frame_index in zip(lengths, freeze_frame_indices, strict=True):
            if freeze_frame_index is not None:
                values[start + freeze_frame_index : start + length] = 0.0
            start += length

    def global_frame_indices(self, clip_ids: torch.Tensor, local_time_steps: torch.Tensor) -> torch.Tensor:
        starts = self.clip_offsets.index_select(0, clip_ids)
        lengths = self.clip_lengths.index_select(0, clip_ids)
        steps = torch.clamp(local_time_steps, min=0)
        steps = torch.minimum(steps, lengths - 1)
        return starts + steps

    def gather_frames(
        self, global_frame_indices: torch.Tensor, out_device: torch.device | str | None = None
    ) -> tuple[torch.Tensor, ...]:
        output_device = torch.device(out_device) if out_device is not None else self.device
        indices = global_frame_indices.to(device=self.storage_device, dtype=torch.long)
        tensors = (
            self.joint_pos.index_select(0, indices),
            self.joint_vel.index_select(0, indices),
            self.body_pos_w.index_select(0, indices),
            self.body_quat_w.index_select(0, indices),
            self.body_lin_vel_w.index_select(0, indices),
            self.body_ang_vel_w.index_select(0, indices),
            self.body_pos_b.index_select(0, indices),
            self.body_quat_b.index_select(0, indices),
            self.body_lin_vel_b.index_select(0, indices),
            self.body_ang_vel_b.index_select(0, indices),
            self.body_lin_acc_w.index_select(0, indices),
            self.body_ang_acc_w.index_select(0, indices),
        )
        if tensors[0].device != output_device:
            tensors = tuple(t.to(output_device, non_blocking=True) for t in tensors)
        return tensors

    def resample_loaded_motions(self) -> bool:
        if len(self._motion_paths) <= self.max_resident_motions:
            return False
        self._load_resident_paths(self._next_resident_paths())
        return True

    @property
    def has_multiple_resident_sets(self) -> bool:
        """Whether the dataset contains clips outside the resident library."""
        return len(self._motion_paths) > self.max_resident_motions

    def to_dataset_dict(self) -> dict[str, Any]:
        return {
            "fps": self.fps,
            "joint_names": self.joint_names,
            "body_names": self.body_names,
            "local_frame_body_name": self.local_frame_body_name,
            "joint_pos": self.joint_pos,
            "joint_vel": self.joint_vel,
            "body_pos_w": self.body_pos_w,
            "body_quat_w": self.body_quat_w,
            "body_lin_vel_w": self.body_lin_vel_w,
            "body_ang_vel_w": self.body_ang_vel_w,
            "body_pos_b": self.body_pos_b,
            "body_quat_b": self.body_quat_b,
            "body_lin_vel_b": self.body_lin_vel_b,
            "body_ang_vel_b": self.body_ang_vel_b,
            "body_lin_acc_w": self.body_lin_acc_w,
            "body_ang_acc_w": self.body_ang_acc_w,
            "clip_offsets": self.clip_offsets,
            "clip_lengths": self.clip_lengths,
            "clip_weights": self.clip_weights,
            "clip_names": self.clip_names,
            "freeze_frame_indices": self.freeze_frame_indices,
        }
