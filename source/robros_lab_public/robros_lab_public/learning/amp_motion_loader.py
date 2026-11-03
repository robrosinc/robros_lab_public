# Copyright (c) 2021-2026, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


from __future__ import annotations

import numpy as np
import pickle
import torch
import warnings
from collections.abc import Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_REQUIRED_MOTION_DATA_KEYS = {
    "fps",
    "joint_names",
    "body_names",
    "local_frame_body_name",
}

_TIME_SERIES_KEYS = (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
    "body_pos_b",
    "body_quat_b",
    "body_lin_vel_b",
    "body_ang_vel_b",
)
_JOINT_TIME_SERIES_KEYS = ("joint_pos", "joint_vel")
_BODY_TIME_SERIES_KEYS = tuple(key for key in _TIME_SERIES_KEYS if key not in _JOINT_TIME_SERIES_KEYS)


@dataclass(frozen=True)
class AMPTermSpec:
    """Description of a flattened AMP observation term."""

    group_name: str
    name: str
    dim: int
    resolver: dict[str, Any] | None = None

    @property
    def qualified_name(self) -> str:
        """Return a stable fully-qualified key for disambiguating duplicate term names."""
        return f"{self.group_name}.{self.name}"


class AMPMotionLoader:
    """Load expert AMP trajectories from canonical DOF-based PKL clips."""

    def __init__(
        self,
        term_specs: Sequence[AMPTermSpec | dict[str, Any]],
        time_between_frames: float,
        device: str = "cpu",
        motion_dir: str | None = None,
        mjcf_file: str | None = None,
        preload_transitions: bool = True,
        num_preload_transitions: int = 200000,
        num_transition_steps: int = 2,
    ) -> None:
        self.device = device
        self.time_between_frames = float(time_between_frames)
        self.preload_transitions = preload_transitions
        self.num_transition_steps = int(num_transition_steps)
        if self.num_transition_steps < 2:
            raise ValueError(f"AMP num_transition_steps must be at least 2, got {self.num_transition_steps}.")

        self.term_specs = [spec if isinstance(spec, AMPTermSpec) else AMPTermSpec(**spec) for spec in term_specs]
        self.observation_dim = sum(spec.dim for spec in self.term_specs)
        self.transition_dim = self.observation_dim * self.num_transition_steps

        self.trajectories: list[torch.Tensor] = []
        trajectory_lengths: list[float] = []
        trajectory_weight_chunks: list[torch.Tensor] = []

        self.motion_files = self._resolve_motion_files(motion_dir)
        self.motion_dir = str(Path(motion_dir).expanduser().resolve()) if motion_dir is not None else ""
        self.mjcf_file = str(Path(mjcf_file).expanduser().resolve()) if mjcf_file is not None else None
        self.motion_file = self.motion_dir
        self._dataset: dict[str, Any] = {}
        self._named_motion: dict[str, dict[str, Mapping[str, Any]]] = {"joints": {}, "bodies": {}}

        reference_fps: float | None = None
        reference_joint_names: tuple[str, ...] | None = None
        reference_body_names: tuple[str, ...] | None = None
        reference_local_frame_body_name: str | None = None

        for motion_file in self.motion_files:
            self.motion_file = str(motion_file)
            dataset = self._load_motion_dataset(motion_file)
            self._dataset = dataset
            self._named_motion = self._load_named_motion(dataset)

            fps = float(np.asarray(dataset["fps"]).reshape(-1)[0])
            if fps <= 0.0:
                raise ValueError(f"AMP motion dataset '{motion_file}' has invalid fps={fps}.")
            joint_names = tuple(str(name) for name in np.asarray(dataset["joint_names"]).reshape(-1).tolist())
            body_names = tuple(str(name) for name in np.asarray(dataset["body_names"]).reshape(-1).tolist())
            local_frame_body_name = str(np.asarray(dataset["local_frame_body_name"]).reshape(-1)[0])

            if reference_fps is None:
                reference_fps = fps
                reference_joint_names = joint_names
                reference_body_names = body_names
                reference_local_frame_body_name = local_frame_body_name
                self.fps = fps
                self.frame_dt = 1.0 / fps
                self.joint_names = joint_names
                self.body_names = body_names
                self.local_frame_body_name = local_frame_body_name
            else:
                if not np.isclose(fps, reference_fps):
                    raise ValueError(f"AMP motion dataset '{motion_file}' fps={fps} does not match {reference_fps}.")
                if joint_names != reference_joint_names:
                    raise ValueError(f"AMP motion dataset '{motion_file}' joint_names do not match.")
                if body_names != reference_body_names:
                    raise ValueError(f"AMP motion dataset '{motion_file}' body_names do not match.")
                if local_frame_body_name != reference_local_frame_body_name:
                    raise ValueError(
                        f"AMP motion dataset '{motion_file}' local_frame_body_name='{local_frame_body_name}' "
                        f"does not match '{reference_local_frame_body_name}'."
                    )

            total_frames = self._infer_total_frames(dataset)
            self._total_frames = total_frames
            if total_frames < 2:
                raise ValueError(f"AMP motion dataset '{motion_file}' must contain at least two frames.")

            clip_offsets = self._load_clip_offsets(dataset, total_frames)
            clip_count = int(clip_offsets.numel() - 1)
            clip_weights = self._load_clip_weights(dataset, clip_count)
            for clip_index in range(clip_count):
                start = int(clip_offsets[clip_index].item())
                end = int(clip_offsets[clip_index + 1].item())
                trajectory = self._build_trajectory_tensor(start, end)
                if trajectory.shape[0] < 2:
                    raise ValueError(
                        f"AMP motion dataset '{motion_file}' clip index {clip_index} must contain at least two frames."
                    )
                self.trajectories.append(trajectory)
                trajectory_lengths.append((trajectory.shape[0] - 1) * self.frame_dt)
            trajectory_weight_chunks.append(clip_weights)

        self.motion_file = self.motion_dir
        trajectory_weights = torch.cat(trajectory_weight_chunks, dim=0)
        self.trajectory_weights = trajectory_weights.to(dtype=torch.float32, device=self.device)
        self.trajectory_weights = torch.clamp(self.trajectory_weights, min=0.0)
        if float(self.trajectory_weights.sum()) <= 0.0:
            raise ValueError(
                f"AMP motion directory '{self.motion_dir}' must contain at least one positive clip weight."
            )
        self.trajectory_weights /= self.trajectory_weights.sum()
        self.trajectory_frame_dt = torch.full(
            (len(self.trajectories),), self.frame_dt, dtype=torch.float32, device=self.device
        )
        self.trajectory_lengths = torch.tensor(trajectory_lengths, dtype=torch.float32, device=self.device)
        self.trajectory_num_frames = torch.tensor(
            [trajectory.shape[0] for trajectory in self.trajectories],
            dtype=torch.long,
            device=self.device,
        )

        self.preloaded_transitions: torch.Tensor | None = None
        if self.preload_transitions:
            self.preloaded_transitions = self._sample_random_transitions(num_preload_transitions)

    def _resolve_motion_files(self, motion_dir: str | None) -> list[Path]:
        if motion_dir is None:
            raise ValueError(
                "AMP motion dataset requires 'motion_dir' to point at a directory of canonical .pkl clips."
            )
        motion_path = Path(motion_dir).expanduser().resolve()
        if motion_path.is_file():
            raise ValueError(f"AMP motion path must be a directory of canonical .pkl clips, got: {motion_path}")
        if not motion_path.is_dir():
            raise FileNotFoundError(f"AMP motion directory does not exist: {motion_path}")
        motion_files = sorted(
            path for path in motion_path.rglob("*.pkl") if path.is_file() and path.name != "metadata.pkl"
        )
        if not motion_files:
            raise FileNotFoundError(f"AMP motion directory does not contain any canonical .pkl files: {motion_path}")
        return motion_files

    def feed_forward_generator(
        self, num_mini_batches: int, mini_batch_size: int
    ) -> Generator[torch.Tensor, None, None]:
        """Yield mini-batches of expert state transitions."""
        for _ in range(num_mini_batches):
            if self.preloaded_transitions is not None:
                indices = torch.randint(0, self.preloaded_transitions.shape[0], (mini_batch_size,), device=self.device)
                yield self.preloaded_transitions[indices]
            else:
                traj_indices = self.sample_trajectory_indices(mini_batch_size)
                times = self.sample_times(traj_indices)
                yield self.get_transition_batch(traj_indices, times)

    def sample_trajectory_indices(self, batch_size: int) -> torch.Tensor:
        """Sample trajectory indices proportional to configured weights."""
        return torch.multinomial(self.trajectory_weights, batch_size, replacement=True)

    def sample_times(self, traj_indices: torch.Tensor) -> torch.Tensor:
        """Sample valid times so the full transition window remains inside each trajectory."""
        transition_horizon = self.time_between_frames * float(self.num_transition_steps - 1)
        max_time = torch.clamp(self.trajectory_lengths[traj_indices] - transition_horizon, min=0.0)
        return torch.rand_like(max_time) * max_time

    def get_transition_batch(self, traj_indices: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Interpolate and flatten a batch of expert transition windows."""
        frames = [
            self.get_observation_batch(traj_indices, times + step * self.time_between_frames)
            for step in range(self.num_transition_steps)
        ]
        return torch.cat(frames, dim=-1)

    def get_observation_batch(self, traj_indices: torch.Tensor, times: torch.Tensor) -> torch.Tensor:
        """Interpolate a batch of AMP observations."""
        batch_size = traj_indices.shape[0]
        observations = torch.zeros(batch_size, self.observation_dim, device=self.device)

        for traj_idx in torch.unique(traj_indices).tolist():
            mask = traj_indices == traj_idx
            trajectory = self.trajectories[traj_idx]
            frame_dt = self.trajectory_frame_dt[traj_idx]
            max_index = trajectory.shape[0] - 1

            position = torch.clamp(times[mask] / frame_dt, min=0.0, max=float(max_index))
            low = torch.floor(position).to(dtype=torch.long)
            high = torch.clamp(low + 1, max=max_index)
            blend = (position - low.to(dtype=position.dtype)).unsqueeze(-1)

            observations[mask] = trajectory[low] * (1.0 - blend) + trajectory[high] * blend

        return observations

    def _sample_random_transitions(self, num_transitions: int) -> torch.Tensor:
        traj_indices = self.sample_trajectory_indices(num_transitions)
        times = self.sample_times(traj_indices)
        return self.get_transition_batch(traj_indices, times)

    def _load_motion_dataset(self, motion_file: Path) -> dict[str, Any]:
        if not motion_file.is_file():
            raise FileNotFoundError(f"AMP motion file does not exist: {motion_file}")
        if motion_file.suffix != ".pkl":
            raise ValueError(f"AMP accepts canonical .pkl clips only, got: {motion_file}")
        if self.mjcf_file is None:
            raise ValueError("AMP canonical PKL loading requires 'mjcf_file' for runtime MJCF FK.")
        from robros_lab_public.utils.motions.motion_io import load_motion_clip
        from robros_lab_public.utils.motions.motion_loader import _FKConfig
        from robros_lab_public.utils.motions.torch_humanoid_batch import Humanoid_Batch

        payload = load_motion_clip(motion_file)
        fk = getattr(self, "_fk", None)
        if fk is None:
            fk = Humanoid_Batch(_FKConfig(Path(self.mjcf_file)), device="cpu")
            self._fk = fk
        root_pos = torch.as_tensor(payload["root_pos"], dtype=torch.float32)
        root_rot = torch.as_tensor(payload["root_rot"], dtype=torch.float32)
        dof_pos = torch.as_tensor(payload["dof_pos"], dtype=torch.float32)
        result = fk.fk_from_dof(root_pos, root_rot, dof_pos, fps=float(payload["fps"]), return_full=True)
        body_pos_w = result.global_translation.numpy()
        body_quat_w = result.global_rotation.numpy()
        body_lin_vel_w = result.global_velocity.numpy()
        body_ang_vel_w = result.global_angular_velocity.numpy()
        anchor_index = fk.body_names.index("base_link") if "base_link" in fk.body_names else 0
        anchor_pos = body_pos_w[:, anchor_index, :]
        anchor_quat = torch.as_tensor(body_quat_w[:, anchor_index, :]).reshape(-1, 4)
        inverse = self._quat_conj_xyzw(anchor_quat)
        body_pos_b = self._quat_rotate_inv_xyzw(
            anchor_quat[:, None, :], torch.as_tensor(body_pos_w - anchor_pos[:, None, :])
        ).numpy()
        body_quat_b = self._quat_mul_xyzw(inverse[:, None, :], torch.as_tensor(body_quat_w)).numpy()
        body_lin_vel_b = self._quat_rotate_inv_xyzw(anchor_quat[:, None, :], torch.as_tensor(body_lin_vel_w)).numpy()
        body_ang_vel_b = self._quat_rotate_inv_xyzw(anchor_quat[:, None, :], torch.as_tensor(body_ang_vel_w)).numpy()
        return {
            "fps": np.asarray(payload["fps"], dtype=np.float32),
            "joint_names": np.asarray(fk.joint_names),
            "body_names": np.asarray(fk.body_names),
            "local_frame_body_name": np.asarray(fk.body_names[anchor_index]),
            "joint_pos": payload["dof_pos"],
            "joint_vel": payload["dof_vel"],
            "body_pos_w": body_pos_w,
            "body_quat_w": body_quat_w,
            "body_lin_vel_w": body_lin_vel_w,
            "body_ang_vel_w": body_ang_vel_w,
            "body_pos_b": body_pos_b,
            "body_quat_b": body_quat_b,
            "body_lin_vel_b": body_lin_vel_b,
            "body_ang_vel_b": body_ang_vel_b,
            "clip_offsets": np.asarray([0, body_pos_w.shape[0]], dtype=np.int64),
        }

    def _load_named_motion(self, data: dict[str, Any]) -> dict[str, dict[str, Mapping[str, Any]]]:
        value = data.get("named_motion")
        if isinstance(value, np.ndarray) and value.dtype == object and value.shape == ():
            value = value.item()
        if value is None:
            value = {}
        if not isinstance(value, Mapping):
            raise TypeError(f"AMP motion dataset '{self.motion_file}' has invalid named_motion payload.")

        joints = value.get("joints", {})
        bodies = value.get("bodies", {})
        if not isinstance(joints, Mapping) or not isinstance(bodies, Mapping):
            raise TypeError(f"AMP motion dataset '{self.motion_file}' has invalid named_motion groups.")
        return {"joints": dict(joints), "bodies": dict(bodies)}

    def _infer_total_frames(self, data: dict[str, Any]) -> int:
        for key in _TIME_SERIES_KEYS:
            if self._has_dense_stream(key):
                return int(np.asarray(data[key]).shape[0])

        for group_name in ("joints", "bodies"):
            group = self._named_motion.get(group_name, {})
            for streams_by_name in group.values():
                if not isinstance(streams_by_name, Mapping):
                    continue
                for values in streams_by_name.values():
                    return int(np.asarray(values).shape[0])

        raise KeyError(
            f"AMP motion dataset '{self.motion_file}' does not contain any dense or name-keyed motion streams."
        )

    def _load_clip_offsets(self, data: dict[str, Any], total_frames: int) -> torch.Tensor:
        if "clip_offsets" in data:
            clip_offsets = np.asarray(data["clip_offsets"], dtype=np.int64).reshape(-1)
            if clip_offsets.size == 0:
                raise ValueError("clip_offsets must contain at least one entry.")
            if clip_offsets[0] != 0:
                clip_offsets = np.concatenate([np.zeros(1, dtype=np.int64), clip_offsets])
            if clip_offsets[-1] != total_frames:
                if clip_offsets[-1] < total_frames:
                    clip_offsets = np.concatenate([clip_offsets, np.asarray([total_frames], dtype=np.int64)])
                else:
                    raise ValueError(
                        f"Invalid clip_offsets in '{self.motion_file}': final offset {clip_offsets[-1]} exceeds {total_frames}."
                    )
        elif "clip_lengths" in data:
            clip_lengths = np.asarray(data["clip_lengths"], dtype=np.int64).reshape(-1)
            clip_offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(clip_lengths, dtype=np.int64)])
        else:
            clip_offsets = np.asarray([0, total_frames], dtype=np.int64)

        if np.any(np.diff(clip_offsets) <= 0):
            raise ValueError(
                f"Invalid clip offsets in '{self.motion_file}': clip boundaries must be strictly increasing."
            )
        if clip_offsets[-1] != total_frames:
            raise ValueError(
                f"Invalid clip metadata in '{self.motion_file}': final clip offset {clip_offsets[-1]} "
                f"does not match total frame count {total_frames}."
            )
        return torch.tensor(clip_offsets, dtype=torch.long, device=self.device)

    def _load_clip_weights(self, data: dict[str, Any], clip_count: int) -> torch.Tensor:
        if "clip_weights" in data:
            clip_weights = torch.tensor(
                np.asarray(data["clip_weights"]).reshape(-1), dtype=torch.float32, device=self.device
            )
            if clip_weights.numel() != clip_count:
                raise ValueError(
                    f"Invalid clip_weights in '{self.motion_file}': got {clip_weights.numel()} weights for {clip_count} clips."
                )
        else:
            clip_weights = torch.ones(clip_count, dtype=torch.float32, device=self.device)
        return clip_weights

    def _build_trajectory_tensor(self, start: int, end: int) -> torch.Tensor:
        num_frames = end - start
        flattened_terms: list[torch.Tensor] = []

        for spec in self.term_specs:
            stream = self._resolve_term_stream(spec, start, end)
            tensor = stream.to(dtype=torch.float32, device=self.device)
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            if tensor.shape[0] != num_frames:
                raise ValueError(
                    f"AMP term '{spec.qualified_name}' expected {num_frames} frames in clip slice [{start}, {end}), "
                    f"got {tensor.shape[0]}."
                )
            tensor = tensor.reshape(num_frames, -1)
            if tensor.shape[1] != spec.dim:
                raise ValueError(
                    f"AMP term '{spec.qualified_name}' expected flattened dim {spec.dim}, got {tensor.shape[1]}."
                )
            flattened_terms.append(tensor)

        if not flattened_terms:
            raise ValueError(f"AMP motion dataset '{self.motion_file}' did not yield any observation terms.")
        return torch.cat(flattened_terms, dim=-1)

    def _resolve_term_stream(self, spec: AMPTermSpec, start: int, end: int) -> torch.Tensor:
        if spec.resolver is not None:
            derived = self._resolve_term_stream_from_dataset(spec, start, end)
            if derived is not None:
                return derived
        else:
            direct = self._resolve_direct_stream(spec, start, end)
            if direct is not None:
                return direct

            derived = self._resolve_term_stream_from_dataset(spec, start, end)
            if derived is not None:
                return derived

        raise KeyError(f"AMP motion dataset '{self.motion_file}' is missing data for term '{spec.qualified_name}'.")

    def _resolve_direct_stream(self, spec: AMPTermSpec, start: int, end: int) -> torch.Tensor | None:
        keys = (spec.qualified_name, spec.name)
        for key in keys:
            value = self._dataset.get(key)
            if value is None:
                continue
            tensor = self._as_tensor(value)
            if tensor.shape[0] != self._total_frames:
                continue
            return tensor[start:end]
        return None

    def _resolve_term_stream_from_dataset(self, spec: AMPTermSpec, start: int, end: int) -> torch.Tensor | None:
        resolver = spec.resolver or {}
        kind = resolver.get("kind")
        if not isinstance(kind, str):
            kind = spec.name

        if kind == "joint_pos":
            return self._resolve_joint_term("joint_pos", resolver, start, end)
        if kind == "joint_vel":
            return self._resolve_joint_term("joint_vel", resolver, start, end)
        if kind == "base_lin_vel":
            return self._slice_body_array("body_lin_vel_b", ["base_link"], start, end)[:, 0, :]
        if kind == "base_ang_vel":
            return self._slice_body_array("body_ang_vel_b", ["base_link"], start, end)[:, 0, :]
        if kind == "projected_gravity":
            root_quat = self._slice_body_array("body_quat_w", ["base_link"], start, end)[:, 0, :]
            gravity_w = torch.zeros(root_quat.shape[0], 3, dtype=torch.float32, device=self.device)
            gravity_w[:, 2] = -1.0
            return self._quat_rotate_inv_xyzw(root_quat, gravity_w)
        if kind == "base_height":
            return self._slice_body_array("body_pos_w", ["base_link"], start, end)[:, 0, 2:3]
        if kind == "link_height":
            body_names = resolver.get("body_names")
            if not isinstance(body_names, list):
                return None
            body_pos_w = self._slice_body_array("body_pos_w", body_names, start, end)
            return body_pos_w[..., 2].reshape(body_pos_w.shape[0], -1)
        if kind == "link_lin_vel":
            return self._resolve_link_vector_in_link_frame("body_lin_vel_w", resolver, start, end)
        if kind == "link_ang_vel":
            return self._resolve_link_vector_in_link_frame("body_ang_vel_w", resolver, start, end)
        if kind == "link_lin_vel_global":
            body_names = resolver.get("body_names")
            if not isinstance(body_names, list):
                return None
            lin_vel_w = self._slice_body_array("body_lin_vel_w", body_names, start, end)
            return lin_vel_w.reshape(lin_vel_w.shape[0], -1)
        if kind == "link_ang_vel_global":
            body_names = resolver.get("body_names")
            if not isinstance(body_names, list):
                return None
            ang_vel_w = self._slice_body_array("body_ang_vel_w", body_names, start, end)
            return ang_vel_w.reshape(ang_vel_w.shape[0], -1)
        if kind == "link_projected_gravity":
            body_names = resolver.get("body_names")
            if not isinstance(body_names, list):
                return None
            body_quat_w = self._slice_body_array("body_quat_w", body_names, start, end)
            gravity_w = torch.zeros(
                body_quat_w.shape[0], body_quat_w.shape[1], 3, dtype=torch.float32, device=self.device
            )
            gravity_w[..., 2] = -1.0
            projected_gravity = self._quat_rotate_inv_xyzw(body_quat_w, gravity_w)
            return projected_gravity.reshape(projected_gravity.shape[0], -1)
        if kind == "link_positions":
            body_names = resolver.get("body_names")
            if not isinstance(body_names, list):
                raise ValueError(
                    f"AMP term '{spec.qualified_name}' with resolver kind 'link_positions' requires body_names."
                )
            anchor_body_name = resolver.get("anchor_body_name")
            body_pos_w = self._slice_body_array("body_pos_w", body_names, start, end)
            anchor_pos_w, anchor_quat_w = self._resolve_anchor_pose(anchor_body_name, start, end)
            anchor_quat = anchor_quat_w.unsqueeze(1).expand(-1, body_pos_w.shape[1], -1)
            rel_pos = self._quat_rotate_inv_xyzw(anchor_quat, body_pos_w - anchor_pos_w.unsqueeze(1))
            return rel_pos.reshape(rel_pos.shape[0], -1)
        if kind == "link_orientations":
            body_names = resolver.get("body_names")
            if not isinstance(body_names, list):
                raise ValueError(
                    f"AMP term '{spec.qualified_name}' with resolver kind 'link_orientations' requires body_names."
                )
            anchor_body_name = resolver.get("anchor_body_name")
            body_quat_w = self._slice_body_array("body_quat_w", body_names, start, end)
            _, anchor_quat_w = self._resolve_anchor_pose(anchor_body_name, start, end)
            anchor_quat = anchor_quat_w.unsqueeze(1).expand(-1, body_quat_w.shape[1], -1)
            rel_quat = self._quat_mul_xyzw(self._quat_conj_xyzw(anchor_quat), body_quat_w)
            return self._normalize_quat(rel_quat).reshape(rel_quat.shape[0], -1)

        return None

    def _resolve_link_vector_in_link_frame(
        self, key: str, resolver: dict[str, Any], start: int, end: int
    ) -> torch.Tensor | None:
        body_names = resolver.get("body_names")
        if not isinstance(body_names, list):
            return None
        vector_w = self._slice_body_array(key, body_names, start, end)
        body_quat_w = self._slice_body_array("body_quat_w", body_names, start, end)
        vector_b = self._quat_rotate_inv_xyzw(body_quat_w, vector_w)
        return vector_b.reshape(vector_b.shape[0], -1)

    def _resolve_joint_term(self, key: str, resolver: dict[str, Any], start: int, end: int) -> torch.Tensor:
        joint_names = resolver.get("joint_names")
        if not isinstance(joint_names, list):
            if self._has_dense_stream(key):
                return self._as_tensor(self._dataset[key])[start:end]
            joint_names = list(self.joint_names)
        return self._slice_joint_array(key, joint_names, start, end)

    def _slice_joint_array(self, key: str, joint_names: list[str], start: int, end: int) -> torch.Tensor:
        joint_index = {name: index for index, name in enumerate(self.joint_names)}
        missing = [name for name in joint_names if name not in joint_index]
        if missing:
            raise KeyError(f"AMP motion dataset '{self.motion_file}' is missing joints required by AMP: {missing}")

        if self._has_dense_stream(key):
            indices = [joint_index[name] for name in joint_names]
            return self._as_tensor(self._dataset[key])[start:end][:, indices]

        stream_by_name = self._get_named_stream_group("joints", key)
        missing_streams = [name for name in joint_names if name not in stream_by_name]
        if missing_streams:
            raise KeyError(
                f"AMP motion dataset '{self.motion_file}' is missing named joint stream '{key}' "
                f"for joints required by AMP: {missing_streams}"
            )
        values = []
        for name in joint_names:
            tensor = self._as_tensor(stream_by_name[name])[start:end]
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            values.append(tensor.reshape(tensor.shape[0], -1))
        return torch.cat(values, dim=-1)

    def _slice_body_array(self, key: str, body_names: list[str], start: int, end: int) -> torch.Tensor:
        body_index = {name: index for index, name in enumerate(self.body_names)}
        missing = [name for name in body_names if name not in body_index]
        if missing:
            raise KeyError(f"AMP motion dataset '{self.motion_file}' is missing bodies required by AMP: {missing}")

        if self._has_dense_stream(key):
            indices = [body_index[name] for name in body_names]
            return self._as_tensor(self._dataset[key])[start:end][:, indices, :]

        stream_by_name = self._get_named_stream_group("bodies", key)
        missing_streams = [name for name in body_names if name not in stream_by_name]
        if missing_streams:
            raise KeyError(
                f"AMP motion dataset '{self.motion_file}' is missing named body stream '{key}' "
                f"for bodies required by AMP: {missing_streams}"
            )
        values = []
        for name in body_names:
            tensor = self._as_tensor(stream_by_name[name])[start:end]
            if tensor.ndim == 1:
                tensor = tensor.unsqueeze(-1)
            values.append(tensor)
        return torch.stack(values, dim=1)

    def _has_dense_stream(self, key: str) -> bool:
        value = self._dataset.get(key)
        return isinstance(value, np.ndarray) and value.dtype != object and value.ndim > 0

    def _get_named_stream_group(self, group_name: str, key: str) -> Mapping[str, Any]:
        group = self._named_motion.get(group_name, {})
        stream_by_name = group.get(key)
        if not isinstance(stream_by_name, Mapping):
            raise KeyError(f"AMP motion dataset '{self.motion_file}' is missing named stream '{group_name}.{key}'.")
        return stream_by_name

    def _resolve_anchor_pose(self, anchor_body_name: Any, start: int, end: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not isinstance(anchor_body_name, str):
            raise ValueError("AMP link-frame terms require an explicit anchor_asset_cfg that resolves to one body.")
        anchor_pos_w = self._slice_body_array("body_pos_w", [anchor_body_name], start, end)[:, 0, :]
        anchor_quat_w = self._slice_body_array("body_quat_w", [anchor_body_name], start, end)[:, 0, :]
        return anchor_pos_w, anchor_quat_w

    def _as_tensor(self, value: Any) -> torch.Tensor:
        return torch.as_tensor(value, dtype=torch.float32, device=self.device)

    def _normalize_quat(self, quat: torch.Tensor, eps: float = 1.0e-8) -> torch.Tensor:
        return quat / torch.clamp(torch.linalg.norm(quat, dim=-1, keepdim=True), min=eps)

    def _quat_conj_xyzw(self, quat: torch.Tensor) -> torch.Tensor:
        output = quat.clone()
        output[..., :3] *= -1.0
        return output

    def _quat_mul_xyzw(self, quat_a: torch.Tensor, quat_b: torch.Tensor) -> torch.Tensor:
        x1, y1, z1, w1 = torch.unbind(quat_a, dim=-1)
        x2, y2, z2, w2 = torch.unbind(quat_b, dim=-1)
        return torch.stack(
            (
                w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            ),
            dim=-1,
        )

    def _quat_rotate_xyzw(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        xyz = quat[..., :3]
        w = quat[..., 3:4]
        cross_1 = 2.0 * torch.cross(xyz, vec, dim=-1)
        return vec + w * cross_1 + torch.cross(xyz, cross_1, dim=-1)

    def _quat_rotate_inv_xyzw(self, quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
        return self._quat_rotate_xyzw(self._quat_conj_xyzw(quat), vec)
