from __future__ import annotations

import copy
import math
import random
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal

import mujoco
import numpy as np
import torch

from mjlab.managers import CommandTerm, CommandTermCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  quat_apply,
  quat_error_magnitude,
  quat_from_euler_xyz,
  quat_inv,
  quat_mul,
  sample_uniform,
  yaw_quat,
)
from mjlab.viewer.debug_visualizer import DebugVisualizer

try:
  from mjlab.utils.wandb import get_wandb_motion_cache_dir

  WANDB_AVAILABLE = True
except ImportError:
  WANDB_AVAILABLE = False
  get_wandb_motion_cache_dir = None  # type: ignore

if TYPE_CHECKING:
  from mjlab.entity import Entity
  from mjlab.envs import ManagerBasedRlEnv

_DESIRED_FRAME_COLORS = ((1.0, 0.5, 0.5), (0.5, 1.0, 0.5), (0.5, 0.5, 1.0))


class MotionLoader:
  def __init__(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    data = np.load(motion_file)
    self._load_from_data(data, body_indexes, device)

  @classmethod
  def from_data(
    cls, data: dict[str, np.ndarray], body_indexes: torch.Tensor, device: str = "cpu"
  ) -> "MotionLoader":
    """Create a MotionLoader from pre-loaded numpy data.

    Args:
      data: Dictionary of numpy arrays (e.g., from np.load)
      body_indexes: Body indices to extract from motion data
      device: Device to load tensors on

    Returns:
      MotionLoader instance
    """
    instance = cls.__new__(cls)
    instance._load_from_data(data, body_indexes, device)
    return instance

  def _load_from_data(
    self, data: dict[str, np.ndarray], body_indexes: torch.Tensor, device: str = "cpu"
  ) -> None:
    """Load motion data from numpy arrays. Handles both 2D (num_frames, ...) and multi-motion formats.

    For multi-motion arrays (ndim >= 3), only the first motion is loaded. Use MultiMotionLoader to handle multiple motions.
    """
    # Check if data is multi-motion format (ndim >= 3) by checking the first array
    joint_pos = data["joint_pos"]
    is_multi_motion = joint_pos.ndim >= 3

    if is_multi_motion:
      # For multi-motion data, take the first motion (will be handled by MultiMotionLoader)
      joint_pos = joint_pos[0]
      joint_vel = data["joint_vel"][0]
      body_pos_w = data["body_pos_w"][0]
      body_quat_w = data["body_quat_w"][0]
      body_lin_vel_w = data["body_lin_vel_w"][0]
      body_ang_vel_w = data["body_ang_vel_w"][0]
    else:
      # For 2D data, use as-is
      joint_pos = data["joint_pos"]
      joint_vel = data["joint_vel"]
      body_pos_w = data["body_pos_w"]
      body_quat_w = data["body_quat_w"]
      body_lin_vel_w = data["body_lin_vel_w"]
      body_ang_vel_w = data["body_ang_vel_w"]

    self.joint_pos = torch.tensor(joint_pos, dtype=torch.float32, device=device)
    self.joint_vel = torch.tensor(joint_vel, dtype=torch.float32, device=device)
    self._body_pos_w = torch.tensor(body_pos_w, dtype=torch.float32, device=device)
    self._body_quat_w = torch.tensor(body_quat_w, dtype=torch.float32, device=device)
    self._body_lin_vel_w = torch.tensor(
      body_lin_vel_w, dtype=torch.float32, device=device
    )
    self._body_ang_vel_w = torch.tensor(
      body_ang_vel_w, dtype=torch.float32, device=device
    )
    self._body_indexes = body_indexes
    self.time_step_total = self.joint_pos.shape[0]

    if "joint_pd_targets" in data:
      pd_targets = data["joint_pd_targets"]
      if is_multi_motion:
        pd_targets = pd_targets[0]
      self._joint_pd_targets = torch.tensor(
        pd_targets, dtype=torch.float32, device=device
      )
    else:
      self._joint_pd_targets = None

  @property
  def body_pos_w(self) -> torch.Tensor:
    return self._body_pos_w[:, self._body_indexes]

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self._body_quat_w[:, self._body_indexes]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self._body_lin_vel_w[:, self._body_indexes]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self._body_ang_vel_w[:, self._body_indexes]

  @property
  def joint_pd_targets(self) -> torch.Tensor | None:
    """PD targets from motion data. Shape: (time_step_total, num_joints)"""
    if not hasattr(self, "_joint_pd_targets") or self._joint_pd_targets is None:
      return None
    return self._joint_pd_targets


class MultiMotionLoader:
  """Loader for multiple motion files that can be selected per environment."""

  def __init__(
    self,
    motion_dir: str,
    motion_name_pattern: list[str],
    body_indexes: torch.Tensor,
    device: str = "cpu",
    pad_to_frames: int | None = None,
  ) -> None:
    """Initialize multi-motion loader.

    Args:
      motion_dir: Base directory containing trajectory subdirectories
      motion_name_pattern: List of regex patterns to match trajectory names (e.g., ".*" matches all).
      body_indexes: Body indices to extract from motion data
      device: Device to load tensors on
      pad_to_frames: Optional frame count to pad all motions to a multiple of (for fixed-length clips)
    """
    motion_dir_path = Path(motion_dir)
    if not motion_dir_path.exists():
      raise ValueError(f"Motion directory does not exist: {motion_dir}")

    # Find all trajectory directories
    traj_dirs = []
    for pattern in motion_name_pattern:
      regex = re.compile(pattern)
      for d in motion_dir_path.iterdir():
        # print(f"[INFO] Checking directory: {d}")
        # print(f"[INFO] Using pattern: {pattern}")
        if not d.is_dir() or not regex.match(d.name):
          continue
        # Check if motion.npz exists directly in this directory
        motion_file = d / "motion.npz"
        if motion_file.exists():
          traj_dirs.append((d, motion_file))
          # print(f"[INFO] Found motion file: {motion_file}")
        else:
          # Check for motion.npz in subdirectories (handles nested structure from wandb downloads)
          for motion_candidate in d.rglob("motion.npz"):
            # motion_name = motion_candidate.parent.name
            traj_dirs.append((motion_candidate.parent, motion_candidate))
            # print(f"[INFO] Found motion file: {motion_candidate}")
            # break  # Only take the first match per directory

    if not traj_dirs:
      raise ValueError(
        f"No matching trajectories found in {motion_dir} with patterns {motion_name_pattern}"
      )

    # Remove duplicates and sort for reproducibility
    traj_dirs = sorted(set(traj_dirs), key=lambda x: x[0].name)

    # Load all motions
    self.motions: list[MotionLoader] = []
    self.traj_names: list[str] = []
    motion_file_summary: list[
      tuple[str, int]
    ] = []  # (motion_file_name, num_trajectories)
    for traj_dir, motion_file in traj_dirs:
      try:
        # Load motions from file (handles both single and multi-motion files)
        motions_from_file = self._load_motions_from_file(
          str(motion_file), body_indexes, device=device
        )
        for i, motion in enumerate[MotionLoader](motions_from_file):
          self.motions.append(motion)
          # For multi-motion files, append index to trajectory name
          if len(motions_from_file) > 1:
            self.traj_names.append(f"{traj_dir.name}_{i}")
          else:
            self.traj_names.append(traj_dir.name)
        motion_file_summary.append((traj_dir.name, len(motions_from_file)))
      except Exception as e:
        print(f"Warning: Failed to load motion from {motion_file}: {e}")
        continue

    # Print summary of loaded motions
    print(
      f"[INFO] Loaded {len(motion_file_summary)} motion file(s) with {len(self.motions)} total trajectory(ies)"
    )
    for motion_file_name, num_trajectories in motion_file_summary:
      print(
        f"[INFO]   Motion file '{motion_file_name}': {num_trajectories} trajectory(ies)"
      )

    if not self.motions:
      raise ValueError(f"No valid motions loaded from {motion_dir}")
    self.num_motions = len(self.motions)
    self.device = device
    self._body_indexes = body_indexes

    # Store number of trajectories per motion file and starting indices
    # This is used to select the first (best) trajectory for each motion file
    self.motion_file_trajectory_counts = [
      num_trajectories for _, num_trajectories in motion_file_summary
    ]
    self.num_motion_files = len(self.motion_file_trajectory_counts)
    self.motion_file_names = [name for name, _ in motion_file_summary]

    # Store max time steps for each motion
    self.time_step_totals = torch.tensor(
      [m.time_step_total for m in self.motions], dtype=torch.long, device=device
    )
    # Base maximum (longest motion)
    self.max_time_steps = int(self.time_step_totals.max().item())
    # Optionally pad the time dimension to a multiple of pad_to_frames (useful
    # when downstream code segments trajectories into fixed-length clips).
    if pad_to_frames is not None and pad_to_frames > 0:
      n = math.ceil(self.max_time_steps / pad_to_frames)
      self.max_time_steps = int(n * pad_to_frames)

    self._has_pd_targets = any(m.joint_pd_targets is not None for m in self.motions)

    # Get shapes from first motion to determine tensor dimensions
    first_motion = self.motions[0]
    num_bodies = len(body_indexes)
    num_joints = first_motion.joint_pos.shape[1]

    # Create batched tensors by padding all motions to max_time_steps
    # Shape: (num_motions, max_time_steps, ...)
    self._batched_joint_pos = torch.zeros(
      self.num_motions,
      self.max_time_steps,
      num_joints,
      dtype=torch.float32,
      device=device,
    )
    self._batched_joint_vel = torch.zeros(
      self.num_motions,
      self.max_time_steps,
      num_joints,
      dtype=torch.float32,
      device=device,
    )
    self._batched_body_pos_w = torch.zeros(
      self.num_motions,
      self.max_time_steps,
      num_bodies,
      3,
      dtype=torch.float32,
      device=device,
    )
    self._batched_body_quat_w = torch.zeros(
      self.num_motions,
      self.max_time_steps,
      num_bodies,
      4,
      dtype=torch.float32,
      device=device,
    )
    self._batched_body_lin_vel_w = torch.zeros(
      self.num_motions,
      self.max_time_steps,
      num_bodies,
      3,
      dtype=torch.float32,
      device=device,
    )
    self._batched_body_ang_vel_w = torch.zeros(
      self.num_motions,
      self.max_time_steps,
      num_bodies,
      3,
      dtype=torch.float32,
      device=device,
    )
    self._batched_pd_targets = torch.zeros(
      self.num_motions,
      self.max_time_steps,
      num_joints,
      dtype=torch.float32,
      device=device,
    )

    # Fill batched tensors by copying data from each motion
    for i, motion in enumerate(self.motions):
      t = motion.time_step_total
      self._batched_joint_pos[i, :t] = motion.joint_pos
      self._batched_joint_vel[i, :t] = motion.joint_vel
      self._batched_body_pos_w[i, :t] = motion.body_pos_w
      self._batched_body_quat_w[i, :t] = motion.body_quat_w
      self._batched_body_lin_vel_w[i, :t] = motion.body_lin_vel_w
      self._batched_body_ang_vel_w[i, :t] = motion.body_ang_vel_w

      if self._batched_pd_targets is not None and motion.joint_pd_targets is not None:
        self._batched_pd_targets[i, :t] = motion.joint_pd_targets

  def _load_motions_from_file(
    self, motion_file: str, body_indexes: torch.Tensor, device: str = "cpu"
  ) -> list[MotionLoader]:
    """Load one or more motions from a file.

    Handles both formats:
    - Single motion: (num_frames, ...) -> returns list with one MotionLoader
    - Multiple motions: (num_motions, num_frames, ...) -> returns list with multiple MotionLoaders

    Args:
      motion_file: Path to the motion file
      body_indexes: Body indices to extract from motion data
      device: Device to load tensors on

    Returns:
      List of MotionLoader instances (one per motion in the file)
    """
    data = np.load(motion_file)

    # Check if data is multi-motion format by checking the first array
    # Multi-motion format: first dimension is num_motions (ndim >= 3)
    joint_pos = data["joint_pos"]
    is_multi_motion = joint_pos.ndim >= 3

    if not is_multi_motion:
      # Single motion file: return list with one MotionLoader
      return [MotionLoader.from_data(data, body_indexes, device=device)]

    # Multiple motions file: create one MotionLoader per motion
    num_motions = joint_pos.shape[0]
    motions = []

    for motion_idx in range(num_motions):
      # Extract data for this motion
      motion_data = {}
      for key, value in data.items():
        if value.ndim >= 3 and value.shape[0] == num_motions:
          # Multi-dimensional array with motion dimension: take the motion_idx slice
          motion_data[key] = value[motion_idx]
        else:
          # 2D, 1D, or scalar array: use as-is (shared across motions or metadata)
          motion_data[key] = value

      motion = MotionLoader.from_data(motion_data, body_indexes, device=device)
      motions.append(motion)

    return motions

  def get_motion_data(
    self,
    motion_indices: torch.Tensor,
    time_steps: torch.Tensor,
    field_name: str,
  ) -> torch.Tensor:
    """Get motion data for specified environments using pure tensor operations.

    Args:
      motion_indices: (num_envs,) tensor of motion indices
      time_steps: (num_envs,) tensor of time steps
      field_name: Name of the field to retrieve (e.g., 'joint_pos', 'body_pos_w')

    Returns:
      Tensor of shape (num_envs, ...) with motion data
    """
    if torch.any(motion_indices < 0) or torch.any(motion_indices >= self.num_motions):
      raise ValueError(
        f"motion_indices out of range: min={motion_indices.min()}, max={motion_indices.max()}, num_motions={self.num_motions}"
      )

    # Clamp time steps to valid range for each motion
    max_times = self.time_step_totals[motion_indices] - 1
    time_steps_clamped = torch.minimum(time_steps, max_times)
    time_steps_clamped = torch.clamp(time_steps_clamped, min=0)

    if field_name == "joint_pos":
      return self._batched_joint_pos[motion_indices, time_steps_clamped]
    elif field_name == "joint_vel":
      return self._batched_joint_vel[motion_indices, time_steps_clamped]
    elif field_name == "body_pos_w":
      return self._batched_body_pos_w[motion_indices, time_steps_clamped]
    elif field_name == "body_quat_w":
      return self._batched_body_quat_w[motion_indices, time_steps_clamped]
    elif field_name == "body_lin_vel_w":
      return self._batched_body_lin_vel_w[motion_indices, time_steps_clamped]
    elif field_name == "body_ang_vel_w":
      return self._batched_body_ang_vel_w[motion_indices, time_steps_clamped]
    elif field_name == "joint_pd_targets":
      if self._batched_pd_targets is None:
        num_envs = motion_indices.shape[0]
        return torch.zeros(
          num_envs,
          self._batched_joint_pos.shape[2],
          dtype=torch.float32,
          device=self.device,
        )
      return self._batched_pd_targets[motion_indices, time_steps_clamped]
    elif field_name == "joint_pd_targets":
      if self._batched_pd_targets is None:
        num_envs = motion_indices.shape[0]
        return torch.zeros(
          num_envs,
          self._batched_joint_pos.shape[2],
          dtype=torch.float32,
          device=self.device,
        )
      return self._batched_pd_targets[motion_indices, time_steps_clamped]
    else:
      raise ValueError(f"Unknown field name: {field_name}")

  def get_motion_data_horizon(
    self,
    motion_indices: torch.Tensor,
    time_steps: torch.Tensor,
    field_name: str,
    horizon: int,
  ) -> torch.Tensor:
    """Get motion data for a horizon of future timesteps.

    Args:
      motion_indices: (num_envs,) tensor of motion indices
      time_steps: (num_envs,) tensor of starting time steps
      field_name: Name of the field to retrieve (e.g., 'joint_pos', 'body_pos_w')
      horizon: Number of future timesteps to retrieve

    Returns:
      Tensor of shape (num_envs, horizon, ...) with motion data for timesteps
      [time_steps, time_steps+1, ..., time_steps+horizon-1]
    """
    if horizon <= 0:
      raise ValueError(f"horizon must be > 0, got {horizon}")

    num_envs = motion_indices.shape[0]
    max_times = self.time_step_totals[motion_indices] - 1

    # Create timestep indices for horizon: (num_envs, horizon)
    # Each row is [t, t+1, ..., t+horizon-1] clamped to valid range
    timestep_indices = (
      time_steps[:, None] + torch.arange(horizon, device=self.device)[None, :]
    )
    timestep_indices = torch.minimum(timestep_indices, max_times[:, None])
    timestep_indices = torch.clamp(timestep_indices, min=0)

    # Get data for all timesteps: (num_envs, horizon, ...)
    # We need to index: batched_data[motion_indices[:, None], timestep_indices]
    # This gives us (num_envs, horizon, ...)
    if field_name == "joint_pos":
      return self._batched_joint_pos[motion_indices[:, None], timestep_indices]
    elif field_name == "joint_vel":
      return self._batched_joint_vel[motion_indices[:, None], timestep_indices]
    elif field_name == "body_pos_w":
      return self._batched_body_pos_w[motion_indices[:, None], timestep_indices]
    elif field_name == "body_quat_w":
      return self._batched_body_quat_w[motion_indices[:, None], timestep_indices]
    elif field_name == "body_lin_vel_w":
      return self._batched_body_lin_vel_w[motion_indices[:, None], timestep_indices]
    elif field_name == "body_ang_vel_w":
      return self._batched_body_ang_vel_w[motion_indices[:, None], timestep_indices]
    elif field_name == "joint_pd_targets":
      if self._batched_pd_targets is None:
        return torch.zeros(
          num_envs,
          horizon,
          self._batched_joint_pos.shape[2],
          dtype=torch.float32,
          device=self.device,
        )
      return self._batched_pd_targets[motion_indices[:, None], timestep_indices]
    else:
      raise ValueError(f"Unknown field name: {field_name}")

  def get_time_step_total(self, motion_indices: torch.Tensor) -> torch.Tensor:
    """Get time step totals for specified motion indices."""
    return self.time_step_totals[motion_indices]


class MotionCommand(CommandTerm):
  cfg: MotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.eef_body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.eef_body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.motion = MotionLoader(
      self.cfg.motion_file, self.body_indexes, device=self.device
    )
    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    self.bin_count = int(self.motion.time_step_total // (1 / env.step_dt)) + 1
    self.bin_failed_count = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self._current_bin_failed = torch.zeros(
      self.bin_count, dtype=torch.float, device=self.device
    )
    self.kernel = torch.tensor(
      [self.cfg.adaptive_lambda**i for i in range(self.cfg.adaptive_kernel_size)],
      device=self.device,
    )
    self.kernel = self.kernel / self.kernel.sum()

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    # Sampling metrics
    # - entropy of the sampling distribution
    # - probability of the top-1 sampled action
    # - binary indicator if the top-1 action was sampled
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    # probability mass of the top-1 sampled bin/clip
    self.metrics["sampling_top1_probability"] = torch.zeros(
      self.num_envs, device=self.device
    )
    # normalized clip index (in [0,1]) of the top-1 sampled bin/clip
    self.metrics["sampling_top1_clip_index"] = torch.zeros(
      self.num_envs, device=self.device
    )
    # (no sampling fallback counter — numerical stabilization applied in
    # adaptive sampling rather than falling back or emitting warnings)

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    # Object/contact handling removed: commands no longer track or write
    # object poses or contact indicators. MotionLoader still provides object
    # properties if available, but command classes ignore them.
    self._has_object = False

  @property
  def command(self) -> torch.Tensor:
    return torch.cat([self.joint_pos, self.joint_vel], dim=1)

  @property
  def joint_pos(self) -> torch.Tensor:
    return self.motion.joint_pos[self.time_steps]

  @property
  def joint_vel(self) -> torch.Tensor:
    return self.motion.joint_vel[self.time_steps]

  @property
  def body_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps] + self._env.scene.env_origins[:, None, :]
    )

  @property
  def body_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps]

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps]

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps]

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    return (
      self.motion.body_pos_w[self.time_steps, self.motion_anchor_body_index]
      + self._env.scene.env_origins
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    return self.motion.body_quat_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    return self.motion.body_lin_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    return self.motion.body_ang_vel_w[self.time_steps, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_eef_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.eef_body_indexes]

  @property
  def robot_eef_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.eef_body_indexes]

  @property
  def joint_pd_targets(self) -> torch.Tensor | None:
    """PD targets from motion data. Shape: (time_step_total, num_joints)"""
    if (
      not hasattr(self.motion, "_joint_pd_targets")
      or self.motion.joint_pd_targets is None
    ):
      return None
    return self.motion.joint_pd_targets[self.time_steps]

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )
    # Object tracking removed from MotionCommand._update_metrics

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      current_bin_index = torch.clamp(
        (self.time_steps * self.bin_count) // max(self.motion.time_step_total, 1),
        0,
        self.bin_count - 1,
      )
      fail_bins = current_bin_index[env_ids][episode_failed]
      self._current_bin_failed[:] = torch.bincount(fail_bins, minlength=self.bin_count)

    # Sample.
    sampling_probabilities = (
      self.bin_failed_count + self.cfg.adaptive_uniform_ratio / float(self.bin_count)
    )
    sampling_probabilities = torch.nn.functional.pad(
      sampling_probabilities.unsqueeze(0).unsqueeze(0),
      (0, self.cfg.adaptive_kernel_size - 1),  # Non-causal kernel
      mode="replicate",
    )
    sampling_probabilities = torch.nn.functional.conv1d(
      sampling_probabilities, self.kernel.view(1, 1, -1)
    ).view(-1)

    sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()

    sampled_bins = torch.multinomial(
      sampling_probabilities, len(env_ids), replacement=True
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count
      * (self.motion.time_step_total - 1)
    ).long()

    # Update metrics.
    H = -(sampling_probabilities * (sampling_probabilities + 1e-12).log()).sum()
    H_norm = H / math.log(self.bin_count)
    pmax, imax = sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_probability"][:] = pmax
    self.metrics["sampling_top1_clip_index"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_probability"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_clip_index"][:] = 0.5  # No specific bin preference.

  def _resample_command(self, env_ids: torch.Tensor):
    if self.cfg.sampling_mode == "start":
      self.time_steps[env_ids] = 0
    elif self.cfg.sampling_mode == "uniform":
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self):
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

    if self.cfg.sampling_mode == "adaptive":
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.asset_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:3]] = self.body_pos_w[visualizer.env_idx, 0].cpu().numpy()
      qpos[free_joint_q_adr[3:7]] = (
        self.body_quat_w[visualizer.env_idx, 0].cpu().numpy()
      )
      qpos[joint_q_adr] = self.joint_pos[visualizer.env_idx].cpu().numpy()

      visualizer.add_ghost_mesh(qpos, model=self._ghost_model)

    elif self.cfg.viz.mode == "frames":
      desired_body_pos = self.body_pos_w[visualizer.env_idx].cpu().numpy()
      desired_body_quat = self.body_quat_w[visualizer.env_idx]
      desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

      current_body_pos = self.robot_body_pos_w[visualizer.env_idx].cpu().numpy()
      current_body_quat = self.robot_body_quat_w[visualizer.env_idx]
      current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

      for i, body_name in enumerate(self.cfg.body_names):
        visualizer.add_frame(
          position=desired_body_pos[i],
          rotation_matrix=desired_body_rotm[i],
          scale=0.08,
          label=f"desired_{body_name}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )
        visualizer.add_frame(
          position=current_body_pos[i],
          rotation_matrix=current_body_rotm[i],
          scale=0.12,
          label=f"current_{body_name}",
        )

      desired_anchor_pos = self.anchor_pos_w[visualizer.env_idx].cpu().numpy()
      desired_anchor_quat = self.anchor_quat_w[visualizer.env_idx]
      desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
      visualizer.add_frame(
        position=desired_anchor_pos,
        rotation_matrix=desired_rotation_matrix,
        scale=0.1,
        label="desired_anchor",
        axis_colors=_DESIRED_FRAME_COLORS,
      )

      current_anchor_pos = self.robot_anchor_pos_w[visualizer.env_idx].cpu().numpy()
      current_anchor_quat = self.robot_anchor_quat_w[visualizer.env_idx]
      current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
      visualizer.add_frame(
        position=current_anchor_pos,
        rotation_matrix=current_rotation_matrix,
        scale=0.15,
        label="current_anchor",
      )


@dataclass(kw_only=True)
class MotionCommandCfg(CommandTermCfg):
  motion_file: str
  anchor_body_name: str
  body_names: tuple[str, ...]
  eef_body_names: tuple[str, ...]
  asset_name: str
  class_type: type[CommandTerm] = MotionCommand
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  object_pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  object_velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)


class MultiMotionCommand(CommandTerm):
  """
  Command term for tracking multiple motion trajectories.
  """

  cfg: MultiMotionCommandCfg
  _env: ManagerBasedRlEnv

  def __init__(self, cfg: MultiMotionCommandCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg, env)

    self.robot: Entity = env.scene[cfg.asset_name]
    self.robot_anchor_body_index = self.robot.body_names.index(
      self.cfg.anchor_body_name
    )
    self.motion_anchor_body_index = self.cfg.body_names.index(self.cfg.anchor_body_name)
    self.body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )
    self.eef_body_indexes = torch.tensor(
      self.robot.find_bodies(self.cfg.eef_body_names, preserve_order=True)[0],
      dtype=torch.long,
      device=self.device,
    )

    # Use motion_dir if already set (e.g., from train.py which downloads beforehand)
    # Otherwise, use cached directory if wandb_entity and wandb_project are provided
    motion_dir = self.cfg.motion_dir
    if (
      motion_dir is None
      and self.cfg.wandb_entity is not None
      and self.cfg.wandb_project is not None
    ):
      if not WANDB_AVAILABLE or get_wandb_motion_cache_dir is None:
        raise ImportError(
          "wandb is required to get motion cache directory. Install with: pip install wandb"
        )
      # Get cache directory path (motions should have been downloaded beforehand)
      cache_dir = get_wandb_motion_cache_dir(
        wandb_entity=self.cfg.wandb_entity,
        wandb_project=self.cfg.wandb_project,
      )
      motion_dir = str(cache_dir)

      # Verify cache directory exists and has motion files
      if not cache_dir.exists():
        raise RuntimeError(
          f"Motion cache directory does not exist: {cache_dir}. "
          f"Please download motions beforehand using download_motions_from_wandb() "
          f"or set motion_dir explicitly."
        )

      motion_files = list(cache_dir.rglob("motion.npz"))
      if not motion_files:
        raise RuntimeError(
          f"No motion.npz files found in cache directory: {cache_dir}. "
          f"Please download motions beforehand using download_motions_from_wandb() "
          f"or set motion_dir explicitly."
        )

      print(
        f"[INFO] Using cached motions from: {motion_dir} ({len(motion_files)} motions found)"
      )
    elif motion_dir is not None:
      print(f"[INFO] Using motion directory: {motion_dir}")
    else:
      # motion_dir is None and no wandb info provided - will be handled by MultiMotionLoader
      pass

    # Determine clip length (in frames) for downstream segmentation and for
    # padding the batched tensors to a multiple of this value. We compute this
    # before creating the MultiMotionLoader so the loader can pad its
    # time-dimension accordingly.
    clip_seconds = self.cfg.clip_seconds
    clip_length_frames = max(1, int(round(clip_seconds / env.step_dt)))

    # Set motion_name_pattern to match all if not explicitly set
    if self.cfg.motion_name_pattern == [".*"]:
      self.cfg.motion_name_pattern = [".*"]
    # print("motion name patterns:", self.cfg.motion_name_pattern)
    # Load multiple motions; request padding to clip length so batched
    # tensors align with clips (prevents partial last clip issues).
    self.motion_loader = MultiMotionLoader(
      motion_dir=motion_dir,
      motion_name_pattern=self.cfg.motion_name_pattern,
      body_indexes=self.body_indexes,
      device=self.device,
      pad_to_frames=clip_length_frames,
    )

    # Load trajectory encoder
    if self.cfg.encoder_dir is not None:
      self.trajectory_encoder = torch.jit.load(
        self.cfg.encoder_dir, map_location=self.device
      )
    else:
      self.trajectory_encoder = None

    # Segment trajectories into fixed-time clips and assign initial clips to envs
    # clip_length_frames was computed above and used to pad the batched tensors.

    clip_traj_indices: list[int] = []
    clip_start_frames: list[int] = []
    clip_lengths: list[int] = []

    num_motions = self.motion_loader.num_motions
    # Sanity check on overlap fraction
    assert (
      self.cfg.clip_overlap_fraction >= 0.0 and self.cfg.clip_overlap_fraction < 1.0
    )
    # Iterate over all motions and segment into clips
    for traj_idx in range(num_motions):
      # time_step_totals is a tensor on motion_loader
      T = int(self.motion_loader.time_step_totals[traj_idx].item())
      # Skip motions with no frames
      if T <= 0:
        continue
      nclips = max(1, math.ceil(T / clip_length_frames))
      # Allow randomized overlap between consecutive clips when configured.
      overlap_max = int(round(self.cfg.clip_overlap_fraction * clip_length_frames))
      for k in range(nclips):
        base_start = k * clip_length_frames
        clip_start = base_start
        clip_end = min(base_start + clip_length_frames, T)

        # Determine left/right expansion (integers) according to clip position.
        left = 0
        right = 0
        if overlap_max > 0:
          if nclips == 1 or (k > 0 and k < nclips - 1):
            # single clip: allow bilateral expansion
            # interior clip: expand both sides independently
            left = random.randint(0, overlap_max)
            right = random.randint(0, overlap_max)
          elif k == 0:
            # first clip: only extend the end
            right = random.randint(0, overlap_max)
          elif k == nclips - 1:
            # last clip: only pull the start backward
            left = random.randint(0, overlap_max)

        # Apply shifts and clamp to motion bounds
        clip_start = int(max(0, clip_start - left))
        clip_end = int(min(T, clip_end + right))

        clip_traj_indices.append(traj_idx)
        clip_start_frames.append(clip_start)
        clip_lengths.append(clip_end - clip_start)

    self.clip_traj_indices = torch.tensor(
      clip_traj_indices, dtype=torch.long, device=self.device
    )
    self.clip_start_frames = torch.tensor(
      clip_start_frames, dtype=torch.long, device=self.device
    )
    self.clip_lengths = torch.tensor(clip_lengths, dtype=torch.long, device=self.device)
    self.num_clips = int(self.clip_traj_indices.shape[0])

    # Precompute representative start joint positions and start anchor positions
    # for each clip. These will be used to compute a tracking-error-based
    # adaptive sampling probability (instead of mean joint position / cdist).
    num_joints = int(self.motion_loader._batched_joint_pos.shape[2])
    self.clip_start_joint_pos = torch.zeros(
      self.num_clips, num_joints, device=self.device
    )

    # Per-clip episode/failure/completion counters for metrics
    self._clip_episode_counts = torch.zeros(
      self.num_clips, dtype=torch.float, device=self.device
    )
    self._clip_failure_counts = torch.zeros(
      self.num_clips, dtype=torch.float, device=self.device
    )
    self._clip_completion_counts = torch.zeros(
      self.num_clips, dtype=torch.float, device=self.device
    )
    # Per-clip EMA for tracking error (preferred to unbounded sum/count).
    # EMA is numerically stable and forgets old observations.
    self._clip_error_ema = torch.zeros(
      self.num_clips, dtype=torch.float, device=self.device
    )
    # Last seen step per clip (initialized to -1 meaning never seen).
    self._clip_last_seen = torch.full(
      (self.num_clips,), -1, dtype=torch.long, device=self.device
    )
    # Global step counter used for aging calculations
    self._global_clip_step = 0

    # Assign initial clips to each environment based on assignment mode randomly
    self.clip_indices = torch.randint(
      0, self.num_clips, (self.num_envs,), device=self.device
    )

    # Derive motion_indices and absolute time_steps from clip assignment
    self.motion_indices = self.clip_traj_indices[self.clip_indices]
    self.time_steps = self.clip_start_frames[self.clip_indices].clone()
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # Number of motion files (for adaptive sampling - group trajectories from same file)
    self.num_motion_files = self.motion_loader.num_motion_files

    self.metrics["error_anchor_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_anchor_lin_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_anchor_ang_vel"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["error_body_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_body_rot"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_pos"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["error_joint_vel"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_probability"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["sampling_top1_clip_index"] = torch.zeros(
      self.num_envs, device=self.device
    )
    # (no sampling fallback counter — numerical stabilization applied in
    # adaptive sampling rather than falling back or emitting warnings)

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

  @property
  def command(self) -> torch.Tensor:
    """Get command (joint_pos + joint_vel) for current timestep only."""
    # When horizon=0 or horizon=1, behave exactly like MotionCommand
    horizon = 1
    if horizon <= 1:
      return torch.cat([self.joint_pos, self.joint_vel], dim=1)
    # For horizon > 1, use horizon methods
    return torch.cat(
      [
        self.get_joint_pos_horizon(horizon),
        self.get_joint_vel_horizon(horizon),
      ],
      dim=1,
    ).view(self.num_envs, -1)

  @property
  def joint_pos(self) -> torch.Tensor:
    """Get joint positions for current timestep only."""
    return self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "joint_pos"
    )

  def get_joint_pos_horizon(self, horizon: int) -> torch.Tensor:
    """Get joint positions for horizon timesteps. For observation use."""
    return self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "joint_pos", horizon
    )

  @property
  def joint_vel(self) -> torch.Tensor:
    """Get joint velocities for current timestep only."""
    return self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "joint_vel"
    )

  def get_joint_vel_horizon(self, horizon: int) -> torch.Tensor:
    """Get joint velocities for horizon timesteps. For observation use."""
    return self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "joint_vel", horizon
    )

  @property
  def body_pos_w(self) -> torch.Tensor:
    """Get body positions for current timestep only."""
    return (
      self.motion_loader.get_motion_data(
        self.motion_indices, self.time_steps, "body_pos_w"
      )
      + self._env.scene.env_origins[:, None, :]
    )

  def get_body_pos_w_horizon(self, horizon: int) -> torch.Tensor:
    """Get body positions for horizon timesteps. For observation use."""
    data = self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "body_pos_w", horizon
    )
    # Add env_origins: (num_envs, horizon, num_bodies, 3) + (num_envs, 1, 1, 3)
    return data + self._env.scene.env_origins[:, None, None, :]

  @property
  def body_quat_w(self) -> torch.Tensor:
    """Get body quaternions for current timestep only."""
    return self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "body_quat_w"
    )

  def get_body_quat_w_horizon(self, horizon: int) -> torch.Tensor:
    """Get body quaternions for horizon timesteps. For observation use."""
    return self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "body_quat_w", horizon
    )

  @property
  def body_lin_vel_w(self) -> torch.Tensor:
    """Get body linear velocities for current timestep only."""
    return self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "body_lin_vel_w"
    )

  def get_body_lin_vel_w_horizon(self, horizon: int) -> torch.Tensor:
    """Get body linear velocities for horizon timesteps. For observation use."""
    return self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "body_lin_vel_w", horizon
    )

  @property
  def body_ang_vel_w(self) -> torch.Tensor:
    """Get body angular velocities for current timestep only."""
    return self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "body_ang_vel_w"
    )

  def get_body_ang_vel_w_horizon(self, horizon: int) -> torch.Tensor:
    """Get body angular velocities for horizon timesteps. For observation use."""
    return self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "body_ang_vel_w", horizon
    )

  @property
  def anchor_pos_w(self) -> torch.Tensor:
    """Get anchor position for current timestep only."""
    body_pos = self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "body_pos_w"
    )
    return body_pos[:, self.motion_anchor_body_index] + self._env.scene.env_origins

  def get_anchor_pos_w_horizon(self, horizon: int) -> torch.Tensor:
    """Get anchor position for horizon timesteps. For observation use."""
    body_pos = self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "body_pos_w", horizon
    )
    # body_pos: (num_envs, horizon, num_bodies, 3)
    # Select anchor body: (num_envs, horizon, 3)
    return (
      body_pos[:, :, self.motion_anchor_body_index]
      + self._env.scene.env_origins[:, None, :]
    )

  @property
  def anchor_quat_w(self) -> torch.Tensor:
    """Get anchor quaternion for current timestep only."""
    body_quat = self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "body_quat_w"
    )
    return body_quat[:, self.motion_anchor_body_index]

  def get_anchor_quat_w_horizon(self, horizon: int) -> torch.Tensor:
    """Get anchor quaternion for horizon timesteps. For observation use."""
    body_quat = self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "body_quat_w", horizon
    )
    # body_quat: (num_envs, horizon, num_bodies, 4)
    # Select anchor body: (num_envs, horizon, 4)
    return body_quat[:, :, self.motion_anchor_body_index]

  @property
  def anchor_lin_vel_w(self) -> torch.Tensor:
    """Get anchor linear velocity for current timestep only."""
    body_lin_vel = self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "body_lin_vel_w"
    )
    return body_lin_vel[:, self.motion_anchor_body_index]

  def get_anchor_lin_vel_w_horizon(self, horizon: int) -> torch.Tensor:
    """Get anchor linear velocity for horizon timesteps. For observation use."""
    body_lin_vel = self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "body_lin_vel_w", horizon
    )
    # body_lin_vel: (num_envs, horizon, num_bodies, 3)
    # Select anchor body: (num_envs, horizon, 3)
    return body_lin_vel[:, :, self.motion_anchor_body_index]

  @property
  def anchor_ang_vel_w(self) -> torch.Tensor:
    """Get anchor angular velocity for current timestep only."""
    body_ang_vel = self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "body_ang_vel_w"
    )
    return body_ang_vel[:, self.motion_anchor_body_index]

  def get_anchor_ang_vel_w_horizon(self, horizon: int) -> torch.Tensor:
    """Get anchor angular velocity for horizon timesteps. For observation use."""
    body_ang_vel = self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "body_ang_vel_w", horizon
    )
    # body_ang_vel: (num_envs, horizon, num_bodies, 3)
    # Select anchor body: (num_envs, horizon, 3)
    return body_ang_vel[:, :, self.motion_anchor_body_index]

  @property
  def robot_joint_pos(self) -> torch.Tensor:
    return self.robot.data.joint_pos

  @property
  def robot_joint_vel(self) -> torch.Tensor:
    return self.robot.data.joint_vel

  @property
  def robot_body_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.body_indexes]

  @property
  def robot_body_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.body_indexes]

  @property
  def robot_body_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.body_indexes]

  @property
  def robot_body_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.body_indexes]

  @property
  def robot_anchor_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_lin_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_lin_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_anchor_ang_vel_w(self) -> torch.Tensor:
    return self.robot.data.body_link_ang_vel_w[:, self.robot_anchor_body_index]

  @property
  def robot_eef_pos_w(self) -> torch.Tensor:
    return self.robot.data.body_link_pos_w[:, self.eef_body_indexes]

  @property
  def robot_eef_quat_w(self) -> torch.Tensor:
    return self.robot.data.body_link_quat_w[:, self.eef_body_indexes]

  @property
  def joint_pd_targets(self) -> torch.Tensor | None:
    """PD targets from motion data. Shape: (time_step_total, num_joints)"""
    return self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "joint_pd_targets"
    )

  def _update_metrics(self):
    self.metrics["error_anchor_pos"] = torch.norm(
      self.anchor_pos_w - self.robot_anchor_pos_w, dim=-1
    )
    self.metrics["error_anchor_rot"] = quat_error_magnitude(
      self.anchor_quat_w, self.robot_anchor_quat_w
    )
    self.metrics["error_anchor_lin_vel"] = torch.norm(
      self.anchor_lin_vel_w - self.robot_anchor_lin_vel_w, dim=-1
    )
    self.metrics["error_anchor_ang_vel"] = torch.norm(
      self.anchor_ang_vel_w - self.robot_anchor_ang_vel_w, dim=-1
    )

    self.metrics["error_body_pos"] = torch.norm(
      self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_rot"] = quat_error_magnitude(
      self.body_quat_relative_w, self.robot_body_quat_w
    ).mean(dim=-1)

    self.metrics["error_body_lin_vel"] = torch.norm(
      self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
    ).mean(dim=-1)
    self.metrics["error_body_ang_vel"] = torch.norm(
      self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
    ).mean(dim=-1)

    self.metrics["error_joint_pos"] = torch.norm(
      self.joint_pos - self.robot_joint_pos, dim=-1
    )
    self.metrics["error_joint_vel"] = torch.norm(
      self.joint_vel - self.robot_joint_vel, dim=-1
    )

    # Accumulate per-clip tracking error for adaptive sampling metrics.
    # We define a simple scalar tracking error as a weighted sum of anchor pos
    # error and joint position error. This is a heuristic; it can be adjusted
    # or exposed as config later.

    # Use configurable weights for tracking-error accumulation
    aw = float(getattr(self.cfg, "tracking_anchor_weight", 1.0))
    jw = float(getattr(self.cfg, "tracking_joint_weight", 0.05))
    tracking_error = (
      aw * self.metrics["error_anchor_pos"] + jw * self.metrics["error_joint_pos"]
    )
    # Update per-clip EMA from current batch of envs. We aggregate per-clip
    # mean error across envs that currently map to each clip, then update
    # an exponential moving average so the estimate remains bounded and
    # responsive to recent behavior.
    ones = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
    sums = torch.zeros(self.num_clips, dtype=torch.float, device=self.device)
    counts = torch.zeros(self.num_clips, dtype=torch.float, device=self.device)
    sums.scatter_add_(0, self.clip_indices, tracking_error)
    counts.scatter_add_(0, self.clip_indices, ones)
    nonzero = counts > 0
    if nonzero.any():
      mean_errors = torch.zeros_like(sums)
      mean_errors[nonzero] = sums[nonzero] / counts[nonzero]
      alpha = self.cfg.clip_ema_alpha
      # Update EMA only for clips observed in this batch
      self._clip_error_ema[nonzero] = (1.0 - alpha) * self._clip_error_ema[
        nonzero
      ] + alpha * mean_errors[nonzero]
      # Record last seen step for these clips
      self._clip_last_seen[nonzero] = int(self._global_clip_step)
    # Advance global step counter (used for aging)
    self._global_clip_step += 1

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    """Adaptive sampling using per-clip EMA-based tracking error.

    This implements a simple hard-example mining policy per environment:
    - With probability `hard_mining_prob` (cfg.hard_mining_prob, default 0.5)
      an env samples a "hard" clip from a weighted distribution that prefers
      high-error clips (higher EMA -> higher prob).
    - Otherwise the env samples uniformly at random.

    The per-clip errors come from an EMA (`self._clip_error_ema`) and are
    age-decayed toward the global mean for clips not seen recently.
    """

    # Build effective per-clip error (age-decayed EMA -> higher => harder)
    # If no clip has been seen, known_mean defaults to 0.0
    known_mask = self._clip_last_seen >= 0
    if known_mask.any():
      known_mean = self._clip_error_ema[known_mask].mean()
    else:
      known_mean = torch.tensor(0.0, device=self.device)

    age = (self._global_clip_step - self._clip_last_seen).clamp(min=0).float()
    tau = float(self.cfg.clip_age_tau)
    decay = torch.exp(-age / (tau + 1e-12))
    clip_effective = self._clip_error_ema * decay + known_mean * (1.0 - decay)

    # Normalize effective error to [0,1]
    eps = 1e-8
    cmin = clip_effective.min()
    cmax = clip_effective.max()
    clip_norm = (clip_effective - cmin) / (cmax - cmin + eps)

    # Build hard-example probability distribution (higher clip_norm -> higher prob)
    logits = clip_norm * self.cfg.adaptive_beta
    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e6, neginf=-1e6)
    hard_probs = torch.softmax(logits, dim=0)
    hard_probs = torch.clamp(
      torch.nan_to_num(hard_probs, nan=0.0, posinf=0.0, neginf=0.0), min=0.0
    )
    if not torch.isfinite(hard_probs.sum()):
      eps2 = 1e-12
      hard_probs = hard_probs + eps2
      hard_probs = hard_probs / hard_probs.sum()

    n = len(env_ids)
    device = self.device

    # Per-env decision: whether to do hard mining or uniform sampling
    hard_mask = torch.rand(n, device=device) < self.cfg.hard_mining_prob

    clip_choices = torch.empty(n, dtype=torch.long, device=device)

    # Uniform choices for envs that did not select hard mining
    nonhard_idx = (~hard_mask).nonzero(as_tuple=False).view(-1)
    if nonhard_idx.numel() > 0:
      clip_choices[nonhard_idx] = torch.randint(
        0, self.num_clips, (nonhard_idx.numel(),), device=device
      )

    # Hard choices sampled from weighted distribution
    hard_idx = hard_mask.nonzero(as_tuple=False).view(-1)
    if hard_idx.numel() > 0:
      # multinomial expects 1D probs; sample as many as needed
      sampled = torch.multinomial(
        hard_probs, num_samples=hard_idx.numel(), replacement=True
      )
      clip_choices[hard_idx] = sampled

    # Assign chosen clips and set offsets
    self.clip_indices[env_ids] = clip_choices
    self.motion_indices[env_ids] = self.clip_traj_indices[clip_choices]
    offsets = (
      torch.rand(n, device=device) * self.clip_lengths[clip_choices].float()
    ).long()
    self.time_steps[env_ids] = self.clip_start_frames[clip_choices] + offsets

    # Metrics: entropy/top1 for hard-sampled envs, uniform metrics for others
    H = -(hard_probs * (hard_probs + 1e-12).log()).sum()
    H_norm = H / math.log(float(self.num_clips) + 1e-12)
    pmax, imax = hard_probs.max(dim=0)
    if nonhard_idx.numel() > 0:
      self.metrics["sampling_entropy"][env_ids[nonhard_idx]] = 1.0
      self.metrics["sampling_top1_probability"][env_ids[nonhard_idx]] = 1.0 / float(
        self.num_clips
      )
      self.metrics["sampling_top1_clip_index"][env_ids[nonhard_idx]] = 0.5
    if hard_idx.numel() > 0:
      self.metrics["sampling_entropy"][env_ids[hard_idx]] = H_norm
      self.metrics["sampling_top1_probability"][env_ids[hard_idx]] = pmax
      self.metrics["sampling_top1_clip_index"][env_ids[hard_idx]] = (
        imax.float() / float(self.num_clips)
      )

  def _uniform_sampling(self, env_ids: torch.Tensor):
    # If clips are present, perform uniform CLIP sampling (choose clip then offset)
    clip_choices = torch.randint(0, self.num_clips, (len(env_ids),), device=self.device)
    # assign chosen clips and set random offsets within each clip
    self.clip_indices[env_ids] = clip_choices
    self.motion_indices[env_ids] = self.clip_traj_indices[clip_choices]
    offsets = (
      torch.rand(len(env_ids), device=self.device)
      * self.clip_lengths[clip_choices].float()
    ).long()
    self.time_steps[env_ids] = self.clip_start_frames[clip_choices] + offsets
    # sampling metrics
    self.metrics["sampling_entropy"][env_ids] = 1.0
    self.metrics["sampling_top1_probability"][env_ids] = 1.0 / float(self.num_clips)
    self.metrics["sampling_top1_clip_index"][env_ids] = 0.5
    return

  def _resample_command(self, env_ids: torch.Tensor):
    # Track per-clip statistics before resampling
    resample_clip_indices = self.clip_indices[env_ids]
    resample_time_steps = self.time_steps[env_ids]

    # Count episodes per clip
    ones = torch.ones(len(env_ids), dtype=torch.float, device=self.device)
    self._clip_episode_counts.scatter_add_(0, resample_clip_indices, ones)

    # Track early terminations / completions per clip
    completion_threshold = 0.95
    clip_starts = self.clip_start_frames[resample_clip_indices]
    clip_lens = self.clip_lengths[resample_clip_indices]
    # progress within clip [0,1]
    progress = (resample_time_steps - clip_starts).float() / (clip_lens.float() + 1e-8)
    is_terminated = self._env.termination_manager.terminated[env_ids]
    is_early_termination = is_terminated & (progress < completion_threshold)
    is_completion = progress >= completion_threshold

    failure_counts = is_early_termination.float()
    completion_counts = is_completion.float()
    self._clip_failure_counts.scatter_add_(0, resample_clip_indices, failure_counts)
    self._clip_completion_counts.scatter_add_(
      0, resample_clip_indices, completion_counts
    )

    # Sampling by clip (delegated to helper methods for clarity).
    if self.cfg.sampling_mode == "start":
      # restart at beginning of assigned clip
      self.time_steps[env_ids] = self.clip_start_frames[self.clip_indices[env_ids]]
    elif self.cfg.sampling_mode == "uniform":
      # delegate to uniform sampler (will handle clip assignment when clips exist)
      self._uniform_sampling(env_ids)
    else:
      assert self.cfg.sampling_mode == "adaptive"
      # delegate to adaptive sampler (will perform clip-aware adaptive sampling when configured)
      self._adaptive_sampling(env_ids)

    root_pos = self.body_pos_w[:, 0].clone()
    root_ori = self.body_quat_w[:, 0].clone()
    root_lin_vel = self.body_lin_vel_w[:, 0].clone()
    root_ang_vel = self.body_ang_vel_w[:, 0].clone()

    range_list = [
      self.cfg.pose_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )

    root_pos[env_ids] += rand_samples[:, 0:3]
    orientations_delta = quat_from_euler_xyz(
      rand_samples[:, 3], rand_samples[:, 4], rand_samples[:, 5]
    )
    root_ori[env_ids] = quat_mul(orientations_delta, root_ori[env_ids])
    range_list = [
      self.cfg.velocity_range.get(key, (0.0, 0.0))
      for key in ["x", "y", "z", "roll", "pitch", "yaw"]
    ]
    ranges = torch.tensor(range_list, device=self.device)
    rand_samples = sample_uniform(
      ranges[:, 0], ranges[:, 1], (len(env_ids), 6), device=self.device
    )
    root_lin_vel[env_ids] += rand_samples[:, :3]
    root_ang_vel[env_ids] += rand_samples[:, 3:]

    joint_pos = self.joint_pos.clone()
    joint_vel = self.joint_vel.clone()

    joint_pos += sample_uniform(
      lower=self.cfg.joint_position_range[0],
      upper=self.cfg.joint_position_range[1],
      size=joint_pos.shape,
      device=joint_pos.device,  # type: ignore
    )
    soft_joint_pos_limits = self.robot.data.soft_joint_pos_limits[env_ids]
    joint_pos[env_ids] = torch.clip(
      joint_pos[env_ids], soft_joint_pos_limits[:, :, 0], soft_joint_pos_limits[:, :, 1]
    )
    self.robot.write_joint_state_to_sim(
      joint_pos[env_ids], joint_vel[env_ids], env_ids=env_ids
    )

    root_state = torch.cat(
      [
        root_pos[env_ids],
        root_ori[env_ids],
        root_lin_vel[env_ids],
        root_ang_vel[env_ids],
      ],
      dim=-1,
    )

    # Write robot root state and clear robot state. Object handling removed.
    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)
    self.robot.clear_state(env_ids=env_ids)

  def _update_command(self):
    self.time_steps += 1
    # Check which envs have exceeded their current clip's time steps
    clip_ends = (
      self.clip_start_frames[self.clip_indices] + self.clip_lengths[self.clip_indices]
    )
    env_ids = torch.where(self.time_steps >= clip_ends)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    anchor_pos_w_repeat = self.anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    anchor_quat_w_repeat = self.anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_pos_w_repeat = self.robot_anchor_pos_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )
    robot_anchor_quat_w_repeat = self.robot_anchor_quat_w[:, None, :].repeat(
      1, len(self.cfg.body_names), 1
    )

    delta_pos_w = robot_anchor_pos_w_repeat
    delta_pos_w[..., 2] = anchor_pos_w_repeat[..., 2]
    delta_ori_w = yaw_quat(
      quat_mul(robot_anchor_quat_w_repeat, quat_inv(anchor_quat_w_repeat))
    )

    self.body_quat_relative_w = quat_mul(delta_ori_w, self.body_quat_w)
    self.body_pos_relative_w = delta_pos_w + quat_apply(
      delta_ori_w, self.body_pos_w - anchor_pos_w_repeat
    )

  def get_per_clip_statistics(self) -> dict:
    """Return per-clip aggregated statistics (CPU tensors).

    Keys: episode_counts, failure_counts, completion_counts, avg_error
    """
    # Return EMA and derived effective error (age-decayed) for inspection.
    known_mask = self._clip_last_seen >= 0
    if known_mask.any():
      known_mean = self._clip_error_ema[known_mask].mean()
    else:
      known_mean = torch.tensor(0.0, device=self.device)
    age = (self._global_clip_step - self._clip_last_seen).clamp(min=0).float()
    tau = float(self.cfg.clip_age_tau)
    decay = torch.exp(-age / (tau + 1e-12))
    effective_error = (self._clip_error_ema * decay + known_mean * (1.0 - decay)).cpu()
    return {
      "episode_counts": self._clip_episode_counts.cpu(),
      "failure_counts": self._clip_failure_counts.cpu(),
      "completion_counts": self._clip_completion_counts.cpu(),
      "avg_error_ema": self._clip_error_ema.cpu(),
      "avg_error_effective": effective_error,
      "last_seen": self._clip_last_seen.cpu(),
    }

  def _debug_vis_impl(self, visualizer: DebugVisualizer) -> None:
    """Draw ghost robot or frames based on visualization mode."""
    if self.cfg.viz.mode == "ghost":
      if self._ghost_model is None:
        self._ghost_model = copy.deepcopy(self._env.sim.mj_model)
        self._ghost_model.geom_rgba[:] = self._ghost_color

      entity: Entity = self._env.scene[self.cfg.asset_name]
      indexing = entity.indexing
      free_joint_q_adr = indexing.free_joint_q_adr.cpu().numpy()
      joint_q_adr = indexing.joint_q_adr.cpu().numpy()

      qpos = np.zeros(self._env.sim.mj_model.nq)
      qpos[free_joint_q_adr[0:3]] = self.body_pos_w[visualizer.env_idx, 0].cpu().numpy()
      qpos[free_joint_q_adr[3:7]] = (
        self.body_quat_w[visualizer.env_idx, 0].cpu().numpy()
      )
      qpos[joint_q_adr] = self.joint_pos[visualizer.env_idx].cpu().numpy()

      # Object visualization removed from ghost view.

      visualizer.add_ghost_mesh(qpos, model=self._ghost_model)

    elif self.cfg.viz.mode == "frames":
      desired_body_pos = self.body_pos_w[visualizer.env_idx].cpu().numpy()
      desired_body_quat = self.body_quat_w[visualizer.env_idx]
      desired_body_rotm = matrix_from_quat(desired_body_quat).cpu().numpy()

      current_body_pos = self.robot_body_pos_w[visualizer.env_idx].cpu().numpy()
      current_body_quat = self.robot_body_quat_w[visualizer.env_idx]
      current_body_rotm = matrix_from_quat(current_body_quat).cpu().numpy()

      for i, body_name in enumerate(self.cfg.body_names):
        visualizer.add_frame(
          position=desired_body_pos[i],
          rotation_matrix=desired_body_rotm[i],
          scale=0.08,
          label=f"desired_{body_name}",
          axis_colors=_DESIRED_FRAME_COLORS,
        )
        visualizer.add_frame(
          position=current_body_pos[i],
          rotation_matrix=current_body_rotm[i],
          scale=0.12,
          label=f"current_{body_name}",
        )

      desired_anchor_pos = self.anchor_pos_w[visualizer.env_idx].cpu().numpy()
      desired_anchor_quat = self.anchor_quat_w[visualizer.env_idx]
      desired_rotation_matrix = matrix_from_quat(desired_anchor_quat).cpu().numpy()
      visualizer.add_frame(
        position=desired_anchor_pos,
        rotation_matrix=desired_rotation_matrix,
        scale=0.1,
        label="desired_anchor",
        axis_colors=_DESIRED_FRAME_COLORS,
      )

      current_anchor_pos = self.robot_anchor_pos_w[visualizer.env_idx].cpu().numpy()
      current_anchor_quat = self.robot_anchor_quat_w[visualizer.env_idx]
      current_rotation_matrix = matrix_from_quat(current_anchor_quat).cpu().numpy()
      visualizer.add_frame(
        position=current_anchor_pos,
        rotation_matrix=current_rotation_matrix,
        scale=0.15,
        label="current_anchor",
      )


@dataclass(kw_only=True)
class MultiMotionCommandCfg(CommandTermCfg):
  motion_dir: str = "motions/output"
  motion_name_pattern: list[str] = field(default_factory=lambda: [".*"])
  anchor_body_name: str
  body_names: tuple[str, ...]
  eef_body_names: tuple[str, ...]
  asset_name: str
  encoder_dir: str | None = None
  wandb_entity: str | None = None
  wandb_project: str | None = None
  class_type: type[CommandTerm] = MultiMotionCommand
  pose_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  velocity_range: dict[str, tuple[float, float]] = field(default_factory=dict)
  joint_position_range: tuple[float, float] = (-0.52, 0.52)
  sampling_mode: Literal["adaptive", "uniform", "start"] = "uniform"
  horizon: int = 0
  """Horizon for future motion data. If > 0, properties will return sequences
  of shape (num_envs, horizon, ...) instead of (num_envs, ...). 
  If 0, returns single timestep data as before."""

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)
  # Clip segmentation: length of each clip in seconds and fraction overlap between
  # consecutive clips (0.0 = no overlap, 0.5 = 50% overlap). When > 0 clips may
  # intersect by up to this fraction of clip length (randomized per-trajectory).
  clip_seconds: float = 1.0
  clip_overlap_fraction: float = 0.0  # must be in [0.0, 1.0)
  clip_age_tau: float = 1000.0
  # Hard-example mining probability for adaptive sampling
  hard_mining_prob: float = 0.5
  # EMA alpha for per-clip tracking error estimation
  clip_ema_alpha: float = 0.1
  # Tracking error combination weights used for clip sampling
  tracking_anchor_weight: float = 1.0
  tracking_joint_weight: float = 0.1
  # Adaptive sampling sharpness (higher => prioritize high-error clips)
  adaptive_beta: float = 1.0
