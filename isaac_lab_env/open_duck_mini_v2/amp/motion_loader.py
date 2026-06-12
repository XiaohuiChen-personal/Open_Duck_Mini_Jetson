# Copyright (c) 2022-2026, The Isaac Lab Project Developers (https://github.com/isaac-sim/IsaacLab/blob/main/CONTRIBUTORS.md).
# Copyright (c) 2025, Open Duck Mini Jetson Project.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

"""Reference-motion loading for AMP training.

``MotionLoader`` is a vendored copy of the Isaac Lab template at
``isaaclab_tasks/direct/humanoid_amp/motions/motion_loader.py`` (kept verbatim
so upstream fixes can be diffed in easily). It loads a single NumPy ``.npz``
motion clip and samples interpolated frames at arbitrary times.

``MultiMotionLoader`` is duck-specific: AMP for the command-conditioned task
trains the discriminator on MULTIPLE reference clips (forward / lateral / turn
gaits from the Open Duck Playground library). It holds N ``MotionLoader``
instances and samples across them with probability proportional to clip
duration, so longer clips contribute proportionally more frames to the
discriminator dataset — equivalent to sampling uniformly over the concatenated
timeline of all clips.
"""

from __future__ import annotations

import os

import numpy as np
import torch


class MotionLoader:
    """
    Helper class to load and sample motion data from NumPy-file format.

    Vendored from Isaac Lab:
    ``source/isaaclab_tasks/isaaclab_tasks/direct/humanoid_amp/motions/motion_loader.py``
    """

    def __init__(self, motion_file: str, device: torch.device) -> None:
        """Load a motion file and initialize the internal variables.

        Args:
            motion_file: Motion file path to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If the specified motion file doesn't exist.
        """
        assert os.path.isfile(motion_file), f"Invalid file path: {motion_file}"
        data = np.load(motion_file)

        self.device = device
        self._dof_names = data["dof_names"].tolist()
        self._body_names = data["body_names"].tolist()

        self.dof_positions = torch.tensor(data["dof_positions"], dtype=torch.float32, device=self.device)
        self.dof_velocities = torch.tensor(data["dof_velocities"], dtype=torch.float32, device=self.device)
        self.body_positions = torch.tensor(data["body_positions"], dtype=torch.float32, device=self.device)
        self.body_rotations = torch.tensor(data["body_rotations"], dtype=torch.float32, device=self.device)
        self.body_linear_velocities = torch.tensor(
            data["body_linear_velocities"], dtype=torch.float32, device=self.device
        )
        self.body_angular_velocities = torch.tensor(
            data["body_angular_velocities"], dtype=torch.float32, device=self.device
        )

        self.dt = 1.0 / data["fps"]
        self.num_frames = self.dof_positions.shape[0]
        self.duration = self.dt * (self.num_frames - 1)
        print(f"Motion loaded ({motion_file}): duration: {self.duration} sec, frames: {self.num_frames}")

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names."""
        return self._dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names."""
        return self._body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return len(self._dof_names)

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return len(self._body_names)

    def _interpolate(
        self,
        a: torch.Tensor,
        *,
        b: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Linear interpolation between consecutive values.

        Args:
            a: The first value. Shape is (N, X) or (N, M, X).
            b: The second value. Shape is (N, X) or (N, M, X).
            blend: Interpolation coefficient between 0 (a) and 1 (b).
            start: Indexes to fetch the first value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).
            end: Indexes to fetch the second value. If both, ``start`` and ``end` are specified,
                the first and second values will be fetches from the argument ``a`` (dimension 0).

        Returns:
            Interpolated values. Shape is (N, X) or (N, M, X).
        """
        if start is not None and end is not None:
            return self._interpolate(a=a[start], b=a[end], blend=blend)
        if a.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if a.ndim >= 3:
            blend = blend.unsqueeze(-1)
        return (1.0 - blend) * a + blend * b

    def _slerp(
        self,
        q0: torch.Tensor,
        *,
        q1: torch.Tensor | None = None,
        blend: torch.Tensor | None = None,
        start: np.ndarray | None = None,
        end: np.ndarray | None = None,
    ) -> torch.Tensor:
        """Interpolation between consecutive rotations (Spherical Linear Interpolation).

        Args:
            q0: The first quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            q1: The second quaternion (wxyz). Shape is (N, 4) or (N, M, 4).
            blend: Interpolation coefficient between 0 (q0) and 1 (q1).
            start: Indexes to fetch the first quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).
            end: Indexes to fetch the second quaternion. If both, ``start`` and ``end` are specified,
                the first and second quaternions will be fetches from the argument ``q0`` (dimension 0).

        Returns:
            Interpolated quaternions. Shape is (N, 4) or (N, M, 4).
        """
        if start is not None and end is not None:
            return self._slerp(q0=q0[start], q1=q0[end], blend=blend)
        if q0.ndim >= 2:
            blend = blend.unsqueeze(-1)
        if q0.ndim >= 3:
            blend = blend.unsqueeze(-1)

        qw, qx, qy, qz = 0, 1, 2, 3  # wxyz
        cos_half_theta = (
            q0[..., qw] * q1[..., qw]
            + q0[..., qx] * q1[..., qx]
            + q0[..., qy] * q1[..., qy]
            + q0[..., qz] * q1[..., qz]
        )

        neg_mask = cos_half_theta < 0
        q1 = q1.clone()
        q1[neg_mask] = -q1[neg_mask]
        cos_half_theta = torch.abs(cos_half_theta)
        cos_half_theta = torch.unsqueeze(cos_half_theta, dim=-1)

        half_theta = torch.acos(cos_half_theta)
        sin_half_theta = torch.sqrt(1.0 - cos_half_theta * cos_half_theta)

        ratio_a = torch.sin((1 - blend) * half_theta) / sin_half_theta
        ratio_b = torch.sin(blend * half_theta) / sin_half_theta

        new_q_x = ratio_a * q0[..., qx : qx + 1] + ratio_b * q1[..., qx : qx + 1]
        new_q_y = ratio_a * q0[..., qy : qy + 1] + ratio_b * q1[..., qy : qy + 1]
        new_q_z = ratio_a * q0[..., qz : qz + 1] + ratio_b * q1[..., qz : qz + 1]
        new_q_w = ratio_a * q0[..., qw : qw + 1] + ratio_b * q1[..., qw : qw + 1]

        new_q = torch.cat([new_q_w, new_q_x, new_q_y, new_q_z], dim=len(new_q_w.shape) - 1)
        new_q = torch.where(torch.abs(sin_half_theta) < 0.001, 0.5 * q0 + 0.5 * q1, new_q)
        new_q = torch.where(torch.abs(cos_half_theta) >= 1, q0, new_q)
        return new_q

    def _compute_frame_blend(self, times: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Compute the indexes of the first and second values, as well as the blending time
        to interpolate between them and the given times.

        Args:
            times: Times, between 0 and motion duration, to sample motion values.
                Specified times will be clipped to fall within the range of the motion duration.

        Returns:
            First value indexes, Second value indexes, and blending time between 0 (first value) and 1 (second value).
        """
        phase = np.clip(times / self.duration, 0.0, 1.0)
        index_0 = (phase * (self.num_frames - 1)).round(decimals=0).astype(int)
        index_1 = np.minimum(index_0 + 1, self.num_frames - 1)
        blend = ((times - index_0 * self.dt) / self.dt).round(decimals=5)
        return index_0, index_1, blend

    def sample_times(self, num_samples: int, duration: float | None = None) -> np.ndarray:
        """Sample random motion times uniformly.

        Args:
            num_samples: Number of time samples to generate.
            duration: Maximum motion duration to sample.
                If not defined samples will be within the range of the motion duration.

        Raises:
            AssertionError: If the specified duration is longer than the motion duration.

        Returns:
            Time samples, between 0 and the specified/motion duration.
        """
        duration = self.duration if duration is None else duration
        assert duration <= self.duration, (
            f"The specified duration ({duration}) is longer than the motion duration ({self.duration})"
        )
        return duration * np.random.uniform(low=0.0, high=1.0, size=num_samples)

    def sample(
        self, num_samples: int, times: np.ndarray | None = None, duration: float | None = None
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data.

        Args:
            num_samples: Number of time samples to generate. If ``times`` is defined, this parameter is ignored.
            times: Motion time used for sampling.
                If not defined, motion data will be random sampled uniformly in time.
            duration: Maximum motion duration to sample.
                If not defined, samples will be within the range of the motion duration.
                If ``times`` is defined, this parameter is ignored.

        Returns:
            A tuple containing sampled motion data:
                - DOF positions (with shape (N, num_dofs))
                - DOF velocities (with shape (N, num_dofs))
                - Body positions (with shape (N, num_bodies, 3))
                - Body rotations (with shape (N, num_bodies, 4), as wxyz quaternion)
                - Body linear velocities (with shape (N, num_bodies, 3))
                - Body angular velocities (with shape (N, num_bodies, 3))
        """
        times = self.sample_times(num_samples, duration) if times is None else times
        index_0, index_1, blend = self._compute_frame_blend(times)
        blend = torch.tensor(blend, dtype=torch.float32, device=self.device)

        return (
            self._interpolate(self.dof_positions, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.dof_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_positions, blend=blend, start=index_0, end=index_1),
            self._slerp(self.body_rotations, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_linear_velocities, blend=blend, start=index_0, end=index_1),
            self._interpolate(self.body_angular_velocities, blend=blend, start=index_0, end=index_1),
        )

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names.

        Args:
            dof_names: List of DOFs names.

        Raises:
            AssertionError: If the specified DOFs name doesn't exist.

        Returns:
            List of DOFs indexes.
        """
        indexes = []
        for name in dof_names:
            assert name in self._dof_names, f"The specified DOF name ({name}) doesn't exist: {self._dof_names}"
            indexes.append(self._dof_names.index(name))
        return indexes

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names.

        Args:
            dof_names: List of body names.

        Raises:
            AssertionError: If the specified body name doesn't exist.

        Returns:
            List of body indexes.
        """
        indexes = []
        for name in body_names:
            assert name in self._body_names, f"The specified body name ({name}) doesn't exist: {self._body_names}"
            indexes.append(self._body_names.index(name))
        return indexes


class MultiMotionLoader:
    """Collection of :class:`MotionLoader` instances sampled by clip duration.

    Design intent: the Isaac Lab AMP template binds one environment to exactly
    one motion clip. The command-conditioned duck task instead needs the
    discriminator to accept ANY gait from a library of clips (forward, lateral,
    turning). This class presents an interface compatible with the env's usage
    of ``MotionLoader`` while drawing samples from N clips with probability
    proportional to each clip's duration (uniform over the concatenated
    timeline, so no clip is over-represented per frame).

    All clips must share the same skeleton (``dof_names``/``body_names``) and
    frame rate, since AMP observation frames from different clips are mixed in
    a single discriminator batch.
    """

    def __init__(self, motion_files: list[str], device: torch.device) -> None:
        """Load all motion files and validate that they share one skeleton.

        Args:
            motion_files: List of motion file paths to load.
            device: The device to which to load the data.

        Raises:
            AssertionError: If no motion file is given, or if the loaded clips
                do not share identical DOF names, body names, or frame rate.
        """
        assert len(motion_files) > 0, "MultiMotionLoader requires at least one motion file"
        self.device = device
        self.loaders = [MotionLoader(motion_file=motion_file, device=device) for motion_file in motion_files]

        # All clips must describe the same skeleton at the same frame rate —
        # their frames are interchangeable rows in the discriminator dataset.
        reference = self.loaders[0]
        for loader, motion_file in zip(self.loaders, motion_files):
            assert loader.dof_names == reference.dof_names, (
                f"Motion file ({motion_file}) DOF names {loader.dof_names} do not match "
                f"the first clip's DOF names {reference.dof_names}"
            )
            assert loader.body_names == reference.body_names, (
                f"Motion file ({motion_file}) body names {loader.body_names} do not match "
                f"the first clip's body names {reference.body_names}"
            )
            assert abs(loader.dt - reference.dt) < 1e-9, (
                f"Motion file ({motion_file}) dt ({loader.dt}) does not match the first clip's dt ({reference.dt})"
            )

        # Clip sampling probabilities, proportional to duration.
        durations = np.array([loader.duration for loader in self.loaders], dtype=np.float64)
        self._sampling_probabilities = durations / durations.sum()

    @property
    def num_motions(self) -> int:
        """Number of loaded motion clips."""
        return len(self.loaders)

    @property
    def dof_names(self) -> list[str]:
        """Skeleton DOF names (shared by all clips)."""
        return self.loaders[0].dof_names

    @property
    def body_names(self) -> list[str]:
        """Skeleton rigid body names (shared by all clips)."""
        return self.loaders[0].body_names

    @property
    def num_dofs(self) -> int:
        """Number of skeleton's DOFs."""
        return self.loaders[0].num_dofs

    @property
    def num_bodies(self) -> int:
        """Number of skeleton's rigid bodies."""
        return self.loaders[0].num_bodies

    @property
    def dt(self) -> float:
        """Time step between motion frames (shared by all clips)."""
        return self.loaders[0].dt

    def sample_loader_ids(self, num_samples: int) -> np.ndarray:
        """Sample clip indexes with probability proportional to clip duration.

        Args:
            num_samples: Number of clip indexes to generate.

        Returns:
            Clip indexes (shape (N,), integers in [0, num_motions)).
        """
        return np.random.choice(self.num_motions, size=num_samples, p=self._sampling_probabilities)

    def sample_times(
        self, num_samples: int, loader_ids: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Sample random (clip, time) pairs uniformly over the clip library.

        Args:
            num_samples: Number of samples to generate. Ignored if ``loader_ids`` is defined.
            loader_ids: Clip indexes to sample times for. If not defined, clips are
                sampled with probability proportional to their duration.

        Returns:
            A tuple of (clip indexes, time samples). Each time sample lies within
            the duration of its assigned clip.
        """
        if loader_ids is None:
            loader_ids = self.sample_loader_ids(num_samples)
        times = np.zeros(loader_ids.shape[0])
        for index, loader in enumerate(self.loaders):
            mask = loader_ids == index
            count = int(mask.sum())
            if count:
                times[mask] = loader.sample_times(count)
        return loader_ids, times

    def sample(
        self,
        num_samples: int,
        loader_ids: np.ndarray | None = None,
        times: np.ndarray | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample motion data across the clip library.

        Args:
            num_samples: Number of samples to generate. Ignored if ``loader_ids``
                and ``times`` are defined.
            loader_ids: Clip index for each sample. If not defined, (clip, time)
                pairs are drawn with probability proportional to clip duration.
            times: Motion time for each sample, paired with ``loader_ids``.
                Must be defined if and only if ``loader_ids`` is defined.

        Returns:
            The same 6-tuple as :meth:`MotionLoader.sample`, with rows gathered
            from the clip assigned to each sample:
                - DOF positions (with shape (N, num_dofs))
                - DOF velocities (with shape (N, num_dofs))
                - Body positions (with shape (N, num_bodies, 3))
                - Body rotations (with shape (N, num_bodies, 4), as wxyz quaternion)
                - Body linear velocities (with shape (N, num_bodies, 3))
                - Body angular velocities (with shape (N, num_bodies, 3))
        """
        if loader_ids is None and times is None:
            loader_ids, times = self.sample_times(num_samples)
        assert loader_ids is not None and times is not None, (
            "loader_ids and times must be both defined or both undefined"
        )
        assert loader_ids.shape[0] == times.shape[0], (
            f"loader_ids ({loader_ids.shape[0]}) and times ({times.shape[0]}) must have the same length"
        )

        total = loader_ids.shape[0]
        dof_positions = torch.zeros((total, self.num_dofs), dtype=torch.float32, device=self.device)
        dof_velocities = torch.zeros((total, self.num_dofs), dtype=torch.float32, device=self.device)
        body_positions = torch.zeros((total, self.num_bodies, 3), dtype=torch.float32, device=self.device)
        body_rotations = torch.zeros((total, self.num_bodies, 4), dtype=torch.float32, device=self.device)
        body_linear_velocities = torch.zeros((total, self.num_bodies, 3), dtype=torch.float32, device=self.device)
        body_angular_velocities = torch.zeros((total, self.num_bodies, 3), dtype=torch.float32, device=self.device)

        # Group samples by clip, sample each clip once, scatter rows back.
        for index, loader in enumerate(self.loaders):
            mask = loader_ids == index
            count = int(mask.sum())
            if not count:
                continue
            mask_tensor = torch.as_tensor(mask, device=self.device)
            (
                dof_positions[mask_tensor],
                dof_velocities[mask_tensor],
                body_positions[mask_tensor],
                body_rotations[mask_tensor],
                body_linear_velocities[mask_tensor],
                body_angular_velocities[mask_tensor],
            ) = loader.sample(num_samples=count, times=times[mask])

        return (
            dof_positions,
            dof_velocities,
            body_positions,
            body_rotations,
            body_linear_velocities,
            body_angular_velocities,
        )

    def get_dof_index(self, dof_names: list[str]) -> list[int]:
        """Get skeleton DOFs indexes by DOFs names (delegates to the first clip).

        Args:
            dof_names: List of DOFs names.

        Returns:
            List of DOFs indexes.
        """
        return self.loaders[0].get_dof_index(dof_names)

    def get_body_index(self, body_names: list[str]) -> list[int]:
        """Get skeleton body indexes by body names (delegates to the first clip).

        Args:
            body_names: List of body names.

        Returns:
            List of body indexes.
        """
        return self.loaders[0].get_body_index(body_names)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--file", type=str, required=True, help="Motion file")
    args, _ = parser.parse_known_args()

    motion = MotionLoader(args.file, "cpu")

    print("- number of frames:", motion.num_frames)
    print("- number of DOFs:", motion.num_dofs)
    print("- number of bodies:", motion.num_bodies)
