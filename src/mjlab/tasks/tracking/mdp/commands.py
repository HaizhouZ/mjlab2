from __future__ import annotations

import copy
import math
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

    if "object_pos_w" in data and "object_quat_w" in data:
      object_pos_w = data["object_pos_w"]
      object_quat_w = data["object_quat_w"]
      if is_multi_motion:
        object_pos_w = object_pos_w[0]
        object_quat_w = object_quat_w[0]
      self._object_pos_w = torch.tensor(
        object_pos_w, dtype=torch.float32, device=device
      )
      self._object_quat_w = torch.tensor(
        object_quat_w, dtype=torch.float32, device=device
      )

      if "object_lin_vel_w" in data:
        object_lin_vel_w = data["object_lin_vel_w"]
        if is_multi_motion:
          object_lin_vel_w = object_lin_vel_w[0]
        self._object_lin_vel_w = torch.tensor(
          object_lin_vel_w, dtype=torch.float32, device=device
        )
      else:
        self._object_lin_vel_w = None

      if "object_ang_vel_w" in data:
        object_ang_vel_w = data["object_ang_vel_w"]
        if is_multi_motion:
          object_ang_vel_w = object_ang_vel_w[0]
        self._object_ang_vel_w = torch.tensor(
          object_ang_vel_w, dtype=torch.float32, device=device
        )
      else:
        self._object_ang_vel_w = None
    else:
      self._object_pos_w = None
      delattr(self, "_object_pos_w")  # Remove attribute if not present
      self._object_quat_w = None
      self._object_lin_vel_w = None
      self._object_ang_vel_w = None

    if "contact_indicators" in data:
      contact_indicators = data["contact_indicators"]
      if is_multi_motion:
        contact_indicators = contact_indicators[0]
      self._object_contact = torch.tensor(
        contact_indicators, dtype=torch.bool, device=device
      )
    else:
      self._object_contact = None

    # Load contact_positions if available
    if "contact_positions" in data:
      contact_positions = data["contact_positions"]
      if is_multi_motion:
        contact_positions = contact_positions[0]
      self._contact_positions = torch.tensor(
        contact_positions, dtype=torch.float32, device=device
      )
    else:
      self._contact_positions = None

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

    # Object pose (global/world) at each time step

  @property
  def object_pos_w(self) -> torch.Tensor:
    if not hasattr(self, "_object_pos_w"):
      return torch.zeros(self.time_step_total, 3, device=self._body_pos_w.device)
    return self._object_pos_w

  @property
  def object_quat_w(self) -> torch.Tensor:
    if not hasattr(self, "_object_quat_w"):
      q = torch.zeros(self.time_step_total, 4, device=self._body_pos_w.device)
      q[:, 0] = 1.0
      return q
    return self._object_quat_w

  @property
  def object_lin_vel_w(self) -> torch.Tensor | None:
    if not hasattr(self, "_object_lin_vel_w"):
      return None
    return self._object_lin_vel_w

  @property
  def object_ang_vel_w(self) -> torch.Tensor | None:
    if not hasattr(self, "_object_ang_vel_w"):
      return None
    return self._object_ang_vel_w

  @property
  def object_contact(self) -> torch.Tensor | None:
    """Contact indicator from motion data. Shape: (time_step_total, num_contacts)"""
    if not hasattr(self, "_object_contact") or self._object_contact is None:
      return None
    return self._object_contact

  @property
  def contact_positions(self) -> torch.Tensor | None:
    """Contact positions from motion data. Shape: (time_step_total, num_contacts, 3)"""
    if not hasattr(self, "_contact_positions") or self._contact_positions is None:
      return None
    return self._contact_positions

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
  ) -> None:
    """Initialize multi-motion loader.

    Args:
      motion_dir: Base directory containing trajectory subdirectories
      motion_name_pattern: List of regex patterns to match trajectory names (e.g., ".*" matches all).
      body_indexes: Body indices to extract from motion data
      device: Device to load tensors on
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

    # Calculate starting indices: cumulative sum of trajectory counts
    # e.g., if counts are [3, 2, 5], start_indices are [0, 3, 5]
    start_idx = 0
    self.motion_file_start_indices = []
    for num_trajectories in self.motion_file_trajectory_counts:
      self.motion_file_start_indices.append(start_idx)
      start_idx += num_trajectories
    self.motion_file_start_indices = torch.tensor(
      self.motion_file_start_indices, dtype=torch.long, device=device
    )
    self.motion_file_trajectory_counts_tensor = torch.tensor(
      self.motion_file_trajectory_counts, dtype=torch.long, device=device
    )

    # Create mapping from trajectory index to motion file index
    # e.g., if counts are [3, 2, 5], mapping is [0,0,0, 1,1, 2,2,2,2,2]
    trajectory_to_file = []
    for file_idx, count in enumerate(self.motion_file_trajectory_counts):
      trajectory_to_file.extend([file_idx] * count)
    self.trajectory_to_motion_file = torch.tensor(
      trajectory_to_file, dtype=torch.long, device=device
    )

    # Store max time steps for each motion
    self.time_step_totals = torch.tensor(
      [m.time_step_total for m in self.motions], dtype=torch.long, device=device
    )
    self.max_time_steps = int(self.time_step_totals.max().item())

    # Check if any motion has object data
    self._has_object = any(hasattr(m, "_object_pos_w") for m in self.motions)
    self._has_contact = any(m.object_contact is not None for m in self.motions)
    self._has_contact_positions = any(
      m.contact_positions is not None for m in self.motions
    )
    self._has_pd_targets = any(m.joint_pd_targets is not None for m in self.motions)

    # Get shapes from first motion to determine tensor dimensions
    first_motion = self.motions[0]
    num_bodies = len(body_indexes)
    num_joints = first_motion.joint_pos.shape[1]

    # Determine shapes for optional fields
    num_contacts = 1
    if self._has_contact:
      for motion in self.motions:
        if motion.object_contact is not None:
          num_contacts = (
            motion.object_contact.shape[1] if motion.object_contact.ndim > 1 else 1
          )
          break

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

    # Optional object tensors
    if self._has_object:
      self._batched_object_pos_w = torch.zeros(
        self.num_motions, self.max_time_steps, 3, dtype=torch.float32, device=device
      )
      self._batched_object_quat_w = torch.zeros(
        self.num_motions, self.max_time_steps, 4, dtype=torch.float32, device=device
      )
      self._batched_object_quat_w[:, :, 0] = 1.0  # Initialize with identity quaternion
      self._batched_object_lin_vel_w = torch.zeros(
        self.num_motions, self.max_time_steps, 3, dtype=torch.float32, device=device
      )
      self._batched_object_ang_vel_w = torch.zeros(
        self.num_motions, self.max_time_steps, 3, dtype=torch.float32, device=device
      )
    else:
      self._batched_object_pos_w = None
      self._batched_object_quat_w = None
      self._batched_object_lin_vel_w = None
      self._batched_object_ang_vel_w = None

    # Optional contact tensors
    if self._has_contact:
      self._batched_object_contact = torch.zeros(
        self.num_motions,
        self.max_time_steps,
        num_contacts,
        dtype=torch.bool,
        device=device,
      )
    else:
      self._batched_object_contact = None

    if self._has_contact_positions:
      self._batched_contact_positions = torch.zeros(
        self.num_motions,
        self.max_time_steps,
        num_contacts,
        3,
        dtype=torch.float32,
        device=device,
      )
    else:
      self._batched_contact_positions = None

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

      if self._has_object and self._batched_object_pos_w is not None:
        if hasattr(motion, "_object_pos_w"):
          self._batched_object_pos_w[i, :t] = motion.object_pos_w
          if self._batched_object_quat_w is not None:
            self._batched_object_quat_w[i, :t] = motion.object_quat_w
          if (
            motion.object_lin_vel_w is not None
            and self._batched_object_lin_vel_w is not None
          ):
            self._batched_object_lin_vel_w[i, :t] = motion.object_lin_vel_w
          if (
            motion.object_ang_vel_w is not None
            and self._batched_object_ang_vel_w is not None
          ):
            self._batched_object_ang_vel_w[i, :t] = motion.object_ang_vel_w

      if (
        self._has_contact
        and self._batched_object_contact is not None
        and motion.object_contact is not None
      ):
        self._batched_object_contact[i, :t] = motion.object_contact

      if (
        self._has_contact_positions
        and self._batched_contact_positions is not None
        and motion.contact_positions is not None
      ):
        self._batched_contact_positions[i, :t] = motion.contact_positions

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
    elif field_name == "object_pos_w":
      if self._batched_object_pos_w is None:
        num_envs = motion_indices.shape[0]
        return torch.zeros(num_envs, 3, dtype=torch.float32, device=self.device)
      return self._batched_object_pos_w[motion_indices, time_steps_clamped]
    elif field_name == "object_quat_w":
      if self._batched_object_quat_w is None:
        num_envs = motion_indices.shape[0]
        quat = torch.zeros(num_envs, 4, dtype=torch.float32, device=self.device)
        quat[:, 0] = 1.0
        return quat
      return self._batched_object_quat_w[motion_indices, time_steps_clamped]
    elif field_name == "object_lin_vel_w":
      if self._batched_object_lin_vel_w is None:
        num_envs = motion_indices.shape[0]
        return torch.zeros(num_envs, 3, dtype=torch.float32, device=self.device)
      return self._batched_object_lin_vel_w[motion_indices, time_steps_clamped]
    elif field_name == "object_ang_vel_w":
      if self._batched_object_ang_vel_w is None:
        num_envs = motion_indices.shape[0]
        return torch.zeros(num_envs, 3, dtype=torch.float32, device=self.device)
      return self._batched_object_ang_vel_w[motion_indices, time_steps_clamped]
    elif field_name == "object_contact":
      if self._batched_object_contact is None:
        num_envs = motion_indices.shape[0]
        # Default to 1 contact if not available
        num_contacts = 1
        return torch.zeros(num_envs, num_contacts, dtype=torch.bool, device=self.device)
      return self._batched_object_contact[motion_indices, time_steps_clamped]
    elif field_name == "contact_positions":
      if self._batched_contact_positions is None:
        num_envs = motion_indices.shape[0]
        # Default to 1 contact if not available
        num_contacts = 1
        return torch.zeros(
          num_envs, num_contacts, 3, dtype=torch.float32, device=self.device
        )
      return self._batched_contact_positions[motion_indices, time_steps_clamped]
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
    elif field_name == "object_pos_w":
      if self._batched_object_pos_w is None:
        return torch.zeros(
          num_envs, horizon, 3, dtype=torch.float32, device=self.device
        )
      return self._batched_object_pos_w[motion_indices[:, None], timestep_indices]
    elif field_name == "object_quat_w":
      if self._batched_object_quat_w is None:
        quat = torch.zeros(
          num_envs, horizon, 4, dtype=torch.float32, device=self.device
        )
        quat[:, :, 0] = 1.0
        return quat
      return self._batched_object_quat_w[motion_indices[:, None], timestep_indices]
    elif field_name == "object_lin_vel_w":
      if self._batched_object_lin_vel_w is None:
        return torch.zeros(
          num_envs, horizon, 3, dtype=torch.float32, device=self.device
        )
      return self._batched_object_lin_vel_w[motion_indices[:, None], timestep_indices]
    elif field_name == "object_ang_vel_w":
      if self._batched_object_ang_vel_w is None:
        return torch.zeros(
          num_envs, horizon, 3, dtype=torch.float32, device=self.device
        )
      return self._batched_object_ang_vel_w[motion_indices[:, None], timestep_indices]
    elif field_name == "object_contact":
      if self._batched_object_contact is None:
        num_contacts = 1
        return torch.zeros(
          num_envs, horizon, num_contacts, dtype=torch.bool, device=self.device
        )
      return self._batched_object_contact[motion_indices[:, None], timestep_indices]
    elif field_name == "contact_positions":
      if self._batched_contact_positions is None:
        num_contacts = 1
        return torch.zeros(
          num_envs, horizon, num_contacts, 3, dtype=torch.float32, device=self.device
        )
      return self._batched_contact_positions[motion_indices[:, None], timestep_indices]
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
    self.metrics["sbto_pd_deviation"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    # Object tracking metrics - only if object exists in motion data
    self.object: Entity | None = self._env.scene.entities.get("object")
    self._has_object = hasattr(self.motion, "_object_pos_w")
    if self._has_object:
      self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["error_object_rot"] = torch.zeros(self.num_envs, device=self.device)

    # Contact indicator from motion data
    self._object_contact = (
      self.motion.object_contact
    )  # (time_step_total, num_contacts) or None
    self._contact_positions = (
      self.motion.contact_positions
    )  # (time_step_total, num_contacts, 3) or None
    if self._object_contact is not None:
      # Initialize contact reference arrays
      num_contacts = (
        self._object_contact.shape[1] if self._object_contact.ndim > 1 else 1
      )
      self.ref_object_contact_future = torch.zeros(
        self.num_envs, 1, num_contacts, dtype=torch.bool, device=self.device
      )
      self.ref_object_contact = torch.zeros(
        self.num_envs, num_contacts, dtype=torch.bool, device=self.device
      )
    else:
      self.ref_object_contact_future = None
      self.ref_object_contact = None

    # Contact mismatch counter for termination condition
    self.contact_mismatch_count = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

    # # Contact target positions (reconstructed from offsets in YAML/config)
    # # These will be set in _update_command based on object pose and offsets
    # contact_target_pos_offset = getattr(cfg, "contact_target_pos_offset", None)
    # contact_eef_pos_offset = getattr(cfg, "contact_eef_pos_offset", None)

    # if contact_target_pos_offset is not None:
    #   # Convert list of lists to tensor
    #   if isinstance(contact_target_pos_offset, list):
    #     self.contact_target_pos_offset = torch.tensor(
    #       contact_target_pos_offset, dtype=torch.float32, device=self.device
    #     )  # (num_contacts, 3)
    #   else:
    #     self.contact_target_pos_offset = contact_target_pos_offset
    #   num_contacts = self.contact_target_pos_offset.shape[0]
    #   self.contact_target_pos_w = torch.zeros(
    #     self.num_envs, num_contacts, 3, dtype=torch.float32, device=self.device
    #   )
    # else:
    #   self.contact_target_pos_offset = None
    #   self.contact_target_pos_w = None

    # if contact_eef_pos_offset is not None:
    #   if isinstance(contact_eef_pos_offset, list):
    #     self.contact_eef_pos_offset = torch.tensor(
    #       contact_eef_pos_offset, dtype=torch.float32, device=self.device
    #     )  # (num_contacts, 3)
    #   else:
    #     self.contact_eef_pos_offset = contact_eef_pos_offset
    # else:
    #   self.contact_eef_pos_offset = None

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
  def object_pos_w(self) -> torch.Tensor:
    pos = self.motion.object_pos_w[self.time_steps]
    # Offset by env origins in x,y to align with robot placement
    return pos + self._env.scene.env_origins

  @property
  def object_quat_w(self) -> torch.Tensor:
    return self.motion.object_quat_w[self.time_steps]

  @property
  def object_lin_vel_w(self) -> torch.Tensor | None:
    return (
      self.motion.object_lin_vel_w[self.time_steps]
      if self.motion.object_lin_vel_w is not None
      else None
    )

  @property
  def object_ang_vel_w(self) -> torch.Tensor | None:
    return (
      self.motion.object_ang_vel_w[self.time_steps]
      if self.motion.object_ang_vel_w is not None
      else None
    )

  @property
  def contact_positions(self) -> torch.Tensor | None:
    """Contact positions for current timestep (world frame). Shape: (num_envs, num_contacts, 3)"""
    if self._contact_positions is None:
      return None
    pos = self._contact_positions[self.time_steps]  # (num_envs, num_contacts, 3)
    # Offset by env origins in x,y to align with robot placement
    # env_origins: (num_envs, 3), need to expand to (num_envs, 1, 3) for broadcasting
    return pos + self._env.scene.env_origins[:, None, :]

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

    joint_pos_action_term = self._env.action_manager.get_term("joint_pos")
    applied_pd_actions = joint_pos_action_term._processed_actions  # type: ignore[attr-defined]
    # Compute sum of squared errors
    self.metrics["sbto_pd_deviation"] = torch.norm(
      self.joint_pd_targets - applied_pd_actions, dim=-1
    )
    if self.object is not None:
      # Desired object pose from motion data
      desired_pos = self.object_pos_w  # (N, 3)
      desired_quat = self.object_quat_w  # (N, 4)

      # Actual object pose from simulation
      actual_pos = self.object.data.body_link_pos_w[:, 0]  # (N, 3) root body
      actual_quat = self.object.data.body_link_quat_w[:, 0]  # (N, 4) root body

      # Compute tracking errors
      self.metrics["error_object_pos"] = torch.norm(desired_pos - actual_pos, dim=-1)
      self.metrics["error_object_rot"] = quat_error_magnitude(desired_quat, actual_quat)

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
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count

  def _uniform_sampling(self, env_ids: torch.Tensor):
    self.time_steps[env_ids] = torch.randint(
      0, self.motion.time_step_total, (len(env_ids),), device=self.device
    )
    self.metrics["sampling_entropy"][:] = 1.0  # Maximum entropy for uniform.
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count
    self.metrics["sampling_top1_bin"][:] = 0.5  # No specific bin preference.

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

    # RSI for object
    if self.object is not None and not self.object.data.is_fixed_base:
      object_pos = self.object_pos_w[env_ids]
      object_quat = self.object_quat_w[env_ids]
      object_lin_vel = self.object_lin_vel_w[env_ids]
      object_ang_vel = self.object_ang_vel_w[env_ids]

      transition_phase = self.time_steps[env_ids] < 50

      if transition_phase.any():
        range_list = [
          self.cfg.object_pose_range.get(key, (0.0, 0.0))
          for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        object_pose_ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
          object_pose_ranges[:, 0],
          object_pose_ranges[:, 1],
          (len(env_ids), 6),
          device=self.device,
        )
        # Pose from motion object state (already includes per-env origin offset via object_pos_w)

        object_pos[transition_phase] += rand_samples[transition_phase, 0:3]
        object_quat[transition_phase] = quat_mul(
          quat_from_euler_xyz(
            rand_samples[transition_phase, 3],
            rand_samples[transition_phase, 4],
            rand_samples[transition_phase, 5],
          ),
          object_quat[transition_phase],
        )
      if (~transition_phase).any():
        range_list = [
          self.cfg.object_velocity_range.get(key, (0.0, 0.0))
          for key in ["x", "y", "z", "roll", "pitch", "yaw"]
        ]
        object_velocity_ranges = torch.tensor(range_list, device=self.device)
        rand_samples = sample_uniform(
          object_velocity_ranges[:, 0],
          object_velocity_ranges[:, 1],
          (len(env_ids), 6),
          device=self.device,
        )
        object_lin_vel[~transition_phase] += rand_samples[~transition_phase, :3]
        object_ang_vel[~transition_phase] += rand_samples[~transition_phase, 3:]

      object_state = torch.cat(
        [
          object_pos,
          object_quat,
          object_lin_vel,
          object_ang_vel,
        ],
        dim=-1,
      )

      self.object.write_root_state_to_sim(object_state, env_ids=env_ids)
      self.object.clear_state(env_ids=env_ids)

      # Reset contact mismatch counter
      self.contact_mismatch_count[env_ids] = 0

  def _update_command(self):
    self.time_steps += 1
    env_ids = torch.where(self.time_steps >= self.motion.time_step_total)[0]
    if env_ids.numel() > 0:
      self._resample_command(env_ids)

    # final_time_steps_env_ids = torch.where(
    #   self.time_steps >= self.motion.time_step_total - 100
    # )[0]
    # if final_time_steps_env_ids.numel() > 0 and self.object is not None:
    #   object_state = torch.cat(
    #     [
    #       self.object.data.body_link_pos_w[final_time_steps_env_ids, 0],
    #       self.object.data.body_link_quat_w[final_time_steps_env_ids, 0],
    #       torch.zeros(len(final_time_steps_env_ids), 3, device=self.device),
    #       torch.zeros(len(final_time_steps_env_ids), 3, device=self.device),
    #     ],
    #     dim=-1,
    #   )
    #   self.object.write_root_state_to_sim(
    #     object_state, env_ids=final_time_steps_env_ids
    #   )
    #   self.object.clear_state(env_ids=final_time_steps_env_ids)

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

    # Update contact indicator from motion data
    if self._object_contact is not None:
      # Get contact indicator for current time steps
      # time_steps: (num_envs,), _object_contact: (time_step_total, num_contacts)
      contact_at_timesteps = self._object_contact[
        self.time_steps
      ]  # (num_envs, num_contacts)
      if contact_at_timesteps.ndim == 1:
        contact_at_timesteps = contact_at_timesteps.unsqueeze(1)  # (num_envs, 1)
      # Update future contact (for horizon=1, just current step)
      self.ref_object_contact_future = contact_at_timesteps.unsqueeze(
        1
      )  # (num_envs, 1, num_contacts)
      self.ref_object_contact = contact_at_timesteps  # (num_envs, num_contacts)

  def check_contact_mismatch(self, sensor_names: list[str]) -> torch.Tensor:
    """Check if actual contacts match reference contacts.

    Args:
      sensor_names: List of contact sensor names to check

    Returns:
      Boolean tensor of shape (num_envs,) indicating which environments have mismatched contacts
    """
    if self.ref_object_contact is None:
      return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    contact_indicators = self.ref_object_contact  # (num_envs, num_contacts)
    num_matches = min(contact_indicators.shape[1], len(sensor_names))

    # Collect sensor found tensors (minimal Python loop for object access)
    found_tensors = []
    valid_indices = []
    for i in range(num_matches):
      if sensor_names[i] in self._env.scene.sensors:
        found = self._env.scene.sensors[sensor_names[i]].data.found
        if found is not None:
          found_tensors.append(found)
          valid_indices.append(i)

    if not found_tensors:
      return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    # Process each tensor to extract contact flags using torch operations
    has_contact_list = []
    for found in found_tensors:
      has_contact = (found > 0).any(dim=1) if found.dim() > 1 else (found > 0)
      has_contact_list.append(has_contact)

    # Stack into tensor: (num_valid_sensors, num_envs) -> (num_envs, num_valid_sensors)
    has_contact_stacked = (
      torch.stack(has_contact_list, dim=0).transpose(0, 1)
      if len(has_contact_list) > 1
      else has_contact_list[0].unsqueeze(1)
    )

    # Map to full num_matches shape using advanced indexing
    has_contact_all = torch.zeros(
      self.num_envs, num_matches, device=self.device, dtype=torch.bool
    )
    if valid_indices:
      has_contact_all[:, torch.tensor(valid_indices, device=self.device)] = (
        has_contact_stacked
      )

    # Compare reference and actual contacts: mismatch if not all match
    return ~(
      contact_indicators[:, :num_matches] == has_contact_all[:, :num_matches]
    ).all(dim=1)

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

      if self.object is not None:
        object_free_joint_q_adr = self.object.indexing.free_joint_q_adr.cpu().numpy()
        target_pos = self.object_pos_w[visualizer.env_idx].cpu().numpy()
        target_quat = self.object_quat_w[visualizer.env_idx].cpu().numpy()
        qpos[object_free_joint_q_adr[:3]] = target_pos
        qpos[object_free_joint_q_adr[3:7]] = target_quat

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
    self.object: Entity | None = env.scene.entities.get("object")  # type: ignore[attr-defined]
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

    # Set motion_name_pattern to match all if not explicitly set
    if self.cfg.motion_name_pattern == [".*"]:
      self.cfg.motion_name_pattern = [".*"]
    # print("motion name patterns:", self.cfg.motion_name_pattern)
    # Load multiple motions
    self.motion_loader = MultiMotionLoader(
      motion_dir=motion_dir,
      motion_name_pattern=self.cfg.motion_name_pattern,
      body_indexes=self.body_indexes,
      device=self.device,
    )

    # Load trajectory encoder
    if self.cfg.encoder_dir is not None:
      self.trajectory_encoder = torch.jit.load(
        self.cfg.encoder_dir, map_location=self.device
      )
    else:
      self.trajectory_encoder = None

    # Assign motions to each environment based on assignment mode
    if self.cfg.motion_assignment_mode == "linear":
      # Linear assignment: env 0 -> motion 0, env 1 -> motion 1, etc. (wraps around)
      self.motion_indices = (
        torch.arange(self.num_envs, device=self.device) % self.motion_loader.num_motions
      )
    elif self.cfg.motion_assignment_mode == "best":
      # Use the first trajectory (index 0) of each motion file
      # Distribute across environments, wrapping around if needed
      num_motion_files = len(self.motion_loader.motion_file_start_indices)
      motion_file_indices = (
        torch.arange(self.num_envs, device=self.device) % num_motion_files
      )
      self.motion_indices = self.motion_loader.motion_file_start_indices[
        motion_file_indices
      ]
    else:  # "random" (default)
      # Randomly assign motions to each environment
      self.motion_indices = torch.randint(
        0, self.motion_loader.num_motions, (self.num_envs,), device=self.device
      )

    self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
    self.body_pos_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 3, device=self.device
    )
    self.body_quat_relative_w = torch.zeros(
      self.num_envs, len(cfg.body_names), 4, device=self.device
    )
    self.body_quat_relative_w[:, :, 0] = 1.0

    # Get max time steps for adaptive sampling (use max across all motions)
    max_time_steps = self.motion_loader.time_step_totals.max().item()
    self.bin_count_max = int(max_time_steps // (1 / env.step_dt)) + 1

    # Number of motion files (for adaptive sampling - group trajectories from same file)
    self.num_motion_files = self.motion_loader.num_motion_files

    # Maintain separate failure distributions for each MOTION FILE (not trajectory)
    # This groups trajectories from the same file together for adaptive sampling
    # Shape: (num_motion_files, bin_count)
    self.bin_failed_count = torch.zeros(
      self.num_motion_files,
      self.bin_count_max,
      dtype=torch.float,
      device=self.device,
    )
    self._current_bin_failed = torch.zeros(
      self.num_motion_files,
      self.bin_count_max,
      dtype=torch.float,
      device=self.device,
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
    self.metrics["sbto_pd_deviation"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_entropy"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_prob"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["sampling_top1_bin"] = torch.zeros(self.num_envs, device=self.device)

    # Multi-motion specific metrics (all meaningful when averaged across envs)
    self.metrics["motion_progress"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["motion_failure_rate"] = torch.zeros(self.num_envs, device=self.device)
    self.metrics["motion_completion_rate"] = torch.zeros(
      self.num_envs, device=self.device
    )
    # Global stats (same value for all envs, so averaging = the value itself)
    self.metrics["worst_motion_failure_rate"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["best_motion_failure_rate"] = torch.zeros(
      self.num_envs, device=self.device
    )
    self.metrics["motion_failure_rate_spread"] = torch.zeros(
      self.num_envs, device=self.device
    )

    # Per-motion-file tracking (aggregated statistics)
    # These track stats per motion FILE, grouping all trajectories from same file
    self._motion_file_episode_counts = torch.zeros(
      self.num_motion_files, dtype=torch.float, device=self.device
    )
    self._motion_file_failure_counts = torch.zeros(
      self.num_motion_files, dtype=torch.float, device=self.device
    )
    self._motion_file_completion_counts = torch.zeros(
      self.num_motion_files, dtype=torch.float, device=self.device
    )
    self._motion_file_error_sum = torch.zeros(
      self.num_motion_files, dtype=torch.float, device=self.device
    )
    self._motion_file_error_count = torch.zeros(
      self.num_motion_files, dtype=torch.float, device=self.device
    )
    # Ghost model created lazily on first visualization
    self._ghost_model: mujoco.MjModel | None = None
    self._ghost_color = np.array(cfg.viz.ghost_color, dtype=np.float32)

    # Object tracking metrics - only if object exists in motion data
    self._has_object = self.motion_loader._has_object
    if self._has_object:
      self.metrics["error_object_pos"] = torch.zeros(self.num_envs, device=self.device)
      self.metrics["error_object_rot"] = torch.zeros(self.num_envs, device=self.device)

    # Contact indicator from motion data
    self._has_contact = self.motion_loader._has_contact
    self._has_contact_positions = self.motion_loader._has_contact_positions
    if self._has_contact:
      # Get contact shape from first motion that has contacts
      for motion in self.motion_loader.motions:
        if motion.object_contact is not None:
          num_contacts = (
            motion.object_contact.shape[1] if motion.object_contact.ndim > 1 else 1
          )
          self.ref_object_contact_future = torch.zeros(
            self.num_envs, 1, num_contacts, dtype=torch.bool, device=self.device
          )
          self.ref_object_contact = torch.zeros(
            self.num_envs, num_contacts, dtype=torch.bool, device=self.device
          )
          break
    else:
      self.ref_object_contact_future = None
      self.ref_object_contact = None

    # Contact mismatch counter for termination condition
    self.contact_mismatch_count = torch.zeros(
      self.num_envs, dtype=torch.long, device=self.device
    )

  @property
  def command(self) -> torch.Tensor:
    """Get command (joint_pos + joint_vel) for current timestep only."""
    # When horizon=0 or horizon=1, behave exactly like MotionCommand
    # horizon = self.cfg.horizon
    # self.cfg.horizon = 1
    # horizon =  self.cfg.horizon
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
  def object_pos_w(self) -> torch.Tensor:
    """Get object position for current timestep only."""
    pos = self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "object_pos_w"
    )
    return pos + self._env.scene.env_origins

  def get_object_pos_w_horizon(self, horizon: int) -> torch.Tensor:
    """Get object position for horizon timesteps. For observation use."""
    pos = self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "object_pos_w", horizon
    )
    # pos: (num_envs, horizon, 3)
    return pos + self._env.scene.env_origins[:, None, :]

  @property
  def object_quat_w(self) -> torch.Tensor:
    """Get object quaternion for current timestep only."""
    return self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "object_quat_w"
    )

  def get_object_quat_w_horizon(self, horizon: int) -> torch.Tensor:
    """Get object quaternion for horizon timesteps. For observation use."""
    return self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "object_quat_w", horizon
    )

  @property
  def object_lin_vel_w(self) -> torch.Tensor | None:
    """Get object linear velocity for current timestep only."""
    if getattr(self.motion_loader, "_batched_object_lin_vel_w", None) is None:
      return None
    return self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "object_lin_vel_w"
    )

  def get_object_lin_vel_w_horizon(self, horizon: int) -> torch.Tensor | None:
    """Get object linear velocity for horizon timesteps. For observation use."""
    if getattr(self.motion_loader, "_batched_object_lin_vel_w", None) is None:
      return None
    return self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "object_lin_vel_w", horizon
    )

  @property
  def object_ang_vel_w(self) -> torch.Tensor | None:
    """Get object angular velocity for current timestep only."""
    if getattr(self.motion_loader, "_batched_object_ang_vel_w", None) is None:
      return None
    return self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "object_ang_vel_w"
    )

  def get_object_ang_vel_w_horizon(self, horizon: int) -> torch.Tensor | None:
    """Get object angular velocity for horizon timesteps. For observation use."""
    if getattr(self.motion_loader, "_batched_object_ang_vel_w", None) is None:
      return None
    return self.motion_loader.get_motion_data_horizon(
      self.motion_indices, self.time_steps, "object_ang_vel_w", horizon
    )

  @property
  def contact_positions(self) -> torch.Tensor | None:
    """Contact positions for current timestep (world frame). Shape: (num_envs, num_contacts, 3)"""
    if not self._has_contact_positions:
      return None
    pos = self.motion_loader.get_motion_data(
      self.motion_indices, self.time_steps, "contact_positions"
    )
    return pos + self._env.scene.env_origins[:, None, :]

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

    joint_pos_action_term = self._env.action_manager.get_term("joint_pos")
    applied_pd_actions = joint_pos_action_term._processed_actions  # type: ignore[attr-defined]
    # Compute sum of squared errors
    if self.joint_pd_targets is not None:
      self.metrics["sbto_pd_deviation"] = torch.norm(
        self.joint_pd_targets - applied_pd_actions, dim=-1
      )
    else:
      self.metrics["sbto_pd_deviation"] = torch.zeros(self.num_envs, device=self.device)

    if self._has_object:
      try:
        object: Entity | None = self._env.scene.entities.get("object")  # type: ignore[attr-defined]
        if object is not None:
          desired_pos = self.object_pos_w
          desired_quat = self.object_quat_w
          actual_pos = object.data.body_link_pos_w[:, 0]
          actual_quat = object.data.body_link_quat_w[:, 0]
          self.metrics["error_object_pos"] = torch.norm(
            desired_pos - actual_pos, dim=-1
          )
          self.metrics["error_object_rot"] = quat_error_magnitude(
            desired_quat, actual_quat
          )
        else:
          self.metrics["error_object_pos"] = torch.zeros(
            self.num_envs, device=self.device
          )
          self.metrics["error_object_rot"] = torch.zeros(
            self.num_envs, device=self.device
          )
      except Exception:
        self.metrics["error_object_pos"] = torch.zeros(
          self.num_envs, device=self.device
        )
        self.metrics["error_object_rot"] = torch.zeros(
          self.num_envs, device=self.device
        )

    # Multi-motion specific metrics (per motion FILE, not per trajectory)
    env_time_step_totals = self.motion_loader.get_time_step_total(self.motion_indices)
    self.metrics["motion_progress"] = self.time_steps.float() / env_time_step_totals

    # Map trajectory indices to motion file indices for metrics
    env_file_indices = self.motion_loader.trajectory_to_motion_file[self.motion_indices]

    # Accumulate per-motion-file errors (using anchor pos as representative error)
    anchor_error = self.metrics["error_anchor_pos"]
    self._motion_file_error_sum.scatter_add_(0, env_file_indices, anchor_error)
    self._motion_file_error_count.scatter_add_(
      0, env_file_indices, torch.ones_like(anchor_error)
    )

    # Compute per-motion-file statistics (avoid division by zero)
    file_failure_rate = self._motion_file_failure_counts / (
      self._motion_file_episode_counts + 1e-8
    )
    file_completion_rate = self._motion_file_completion_counts / (
      self._motion_file_episode_counts + 1e-8
    )

    # Per-env metrics: each env gets its motion FILE's rate
    self.metrics["motion_failure_rate"][:] = file_failure_rate[env_file_indices]
    self.metrics["motion_completion_rate"][:] = file_completion_rate[env_file_indices]

    # Global metrics (same for all envs - these average correctly in wandb)
    valid_files = self._motion_file_episode_counts > 0
    if valid_files.any():
      valid_rates = file_failure_rate[valid_files]
      # Worst motion file = highest failure rate (hardest to track)
      self.metrics["worst_motion_failure_rate"][:] = valid_rates.max()
      # Best motion file = lowest failure rate (easiest to track)
      self.metrics["best_motion_failure_rate"][:] = valid_rates.min()
      # Spread = difference between worst and best (large = inconsistent difficulty)
      self.metrics["motion_failure_rate_spread"][:] = (
        valid_rates.max() - valid_rates.min()
      )

  def _adaptive_sampling(self, env_ids: torch.Tensor):
    """
    Adaptive sampling implementation with MOTION FILE level grouping.

    Trajectories from the same motion file are grouped together for adaptive sampling.
    This provides more samples per group when you have many trajectories per file.
    E.g., 30 files × 60 trajectories = 1800 total, but only 30 adaptive sampling groups.

    Failure tracking and sampling probabilities are computed per motion FILE,
    but individual trajectories within a file can still be randomly selected.
    """
    episode_failed = self._env.termination_manager.terminated[env_ids]
    if torch.any(episode_failed):
      # Get trajectory indices and time steps for failed episodes
      failed_env_ids = env_ids[episode_failed]
      failed_traj_indices = self.motion_indices[failed_env_ids]
      failed_time_steps = self.time_steps[failed_env_ids]

      # Map trajectory indices to motion FILE indices
      failed_file_indices = self.motion_loader.trajectory_to_motion_file[
        failed_traj_indices
      ]

      # Get time step totals for each failed env's trajectory
      failed_env_time_step_totals = self.motion_loader.get_time_step_total(
        failed_traj_indices
      )

      # Compute bin indices for failed episodes
      current_bin_index = torch.clamp(
        (failed_time_steps * self.bin_count_max)
        // torch.clamp(failed_env_time_step_totals, min=1),
        0,
        self.bin_count_max - 1,
      )

      # Vectorized update: use motion FILE index (not trajectory index)
      # Create linear indices: file_idx * bin_count + bin_idx
      linear_indices = failed_file_indices * self.bin_count_max + current_bin_index
      # Count occurrences of each (file, bin) pair
      counts = torch.bincount(
        linear_indices, minlength=self.num_motion_files * self.bin_count_max
      )
      # Reshape and add to _current_bin_failed
      self._current_bin_failed += counts.view(
        self.num_motion_files, self.bin_count_max
      ).float()

    # Get motion FILE indices for environments being resampled
    resample_traj_indices = self.motion_indices[env_ids]
    resample_file_indices = self.motion_loader.trajectory_to_motion_file[
      resample_traj_indices
    ]

    # Pre-compute sampling probabilities for all motion FILES (vectorized)
    # Shape: (num_motion_files, bin_count)
    all_sampling_probabilities = (
      self.bin_failed_count
      + self.cfg.adaptive_uniform_ratio / float(self.bin_count_max)
    )
    # Pad for convolution: (num_motion_files, 1, bin_count + kernel_size - 1)
    all_sampling_probabilities_padded = torch.nn.functional.pad(
      all_sampling_probabilities.unsqueeze(1),
      (0, self.cfg.adaptive_kernel_size - 1),
      mode="replicate",
    )
    # Apply convolution to all motion files at once: (num_motion_files, 1, bin_count)
    all_sampling_probabilities = torch.nn.functional.conv1d(
      all_sampling_probabilities_padded, self.kernel.view(1, 1, -1)
    ).squeeze(1)  # (num_motion_files, bin_count)

    # Normalize each motion file's distribution
    all_sampling_probabilities = (
      all_sampling_probabilities / all_sampling_probabilities.sum(dim=1, keepdim=True)
    )

    # Sample bins using the motion FILE's distribution (not trajectory)
    # Get probabilities for each env's motion file: (len(env_ids), bin_count)
    env_sampling_probabilities = all_sampling_probabilities[resample_file_indices]

    # Sample all at once - multinomial supports different distributions per row
    # Shape: (len(env_ids),) - one sample per environment
    sampled_bins = torch.multinomial(
      env_sampling_probabilities, num_samples=1, replacement=True
    ).squeeze(1)

    # Convert sampled bins to time steps for each environment
    # Use the actual trajectory's time_step_total (all trajectories in a file have same length)
    env_time_step_totals = self.motion_loader.get_time_step_total(
      self.motion_indices[env_ids]
    )
    self.time_steps[env_ids] = (
      (sampled_bins + sample_uniform(0.0, 1.0, (len(env_ids),), device=self.device))
      / self.bin_count_max
      * (env_time_step_totals - 1)
    ).long()

    # Compute metrics based on the actual distributions being used for sampling
    # Weight by the number of environments using each motion file
    file_counts = torch.bincount(
      resample_file_indices, minlength=self.num_motion_files
    ).float()
    # Normalize to get weights
    file_weights = file_counts / (file_counts.sum() + 1e-12)

    # Compute weighted average of sampling probabilities
    # all_sampling_probabilities is already normalized per file: (num_motion_files, bin_count)
    weighted_sampling_probabilities = (
      all_sampling_probabilities * file_weights[:, None]
    ).sum(dim=0)
    # Normalize to ensure it sums to 1
    weighted_sampling_probabilities = weighted_sampling_probabilities / (
      weighted_sampling_probabilities.sum() + 1e-12
    )

    H = -(
      weighted_sampling_probabilities * (weighted_sampling_probabilities + 1e-12).log()
    ).sum()
    H_norm = H / math.log(self.bin_count_max)
    pmax, imax = weighted_sampling_probabilities.max(dim=0)
    self.metrics["sampling_entropy"][:] = H_norm
    self.metrics["sampling_top1_prob"][:] = pmax
    self.metrics["sampling_top1_bin"][:] = imax.float() / self.bin_count_max

  def _uniform_sampling(self, env_ids: torch.Tensor):
    # Get time step totals for each env's motion
    env_time_step_totals = self.motion_loader.get_time_step_total(
      self.motion_indices[env_ids]
    )
    # Sample uniformly for each env based on its motion's length
    self.time_steps[env_ids] = (
      torch.rand(len(env_ids), device=self.device) * env_time_step_totals
    ).long()
    self.metrics["sampling_entropy"][:] = 1.0
    self.metrics["sampling_top1_prob"][:] = 1.0 / self.bin_count_max
    self.metrics["sampling_top1_bin"][:] = 0.5

  def _resample_command(self, env_ids: torch.Tensor):
    # Track episode statistics before resampling
    if len(env_ids) > 0:
      resample_traj_indices = self.motion_indices[env_ids]
      resample_time_steps = self.time_steps[env_ids]
      resample_time_step_totals = self.motion_loader.get_time_step_total(
        resample_traj_indices
      )

      # Map trajectory indices to motion FILE indices for tracking
      resample_file_indices = self.motion_loader.trajectory_to_motion_file[
        resample_traj_indices
      ]

      # Count episodes per motion FILE (vectorized scatter)
      ones = torch.ones(len(env_ids), dtype=torch.float, device=self.device)
      self._motion_file_episode_counts.scatter_add_(0, resample_file_indices, ones)

      # Track failures (terminated before reaching near-end of motion)
      # Consider "completion" if within last 5% of motion
      completion_threshold = 0.95
      progress = resample_time_steps.float() / resample_time_step_totals
      is_terminated = self._env.termination_manager.terminated[env_ids]
      is_early_termination = is_terminated & (progress < completion_threshold)
      is_completion = progress >= completion_threshold

      # Accumulate failure counts per motion FILE
      failure_counts = is_early_termination.float()
      self._motion_file_failure_counts.scatter_add_(
        0, resample_file_indices, failure_counts
      )

      # Accumulate completion counts per motion FILE
      completion_counts = is_completion.float()
      self._motion_file_completion_counts.scatter_add_(
        0, resample_file_indices, completion_counts
      )

      self.metrics["motion_progress"][env_ids] = (
        resample_time_steps.float() / resample_time_step_totals
      )

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

    try:
      object: Entity | None = self._env.scene.entities.get("object")  # type: ignore[attr-defined]
    except Exception:
      object: Entity | None = None

    if object is not None and not object.data.is_fixed_base:
      object_pos = self.object_pos_w[env_ids]
      object_quat = self.object_quat_w[env_ids]
      object_lin_vel = (
        self.object_lin_vel_w[env_ids]
        if self.object_lin_vel_w is not None
        else torch.zeros(len(env_ids), 3, device=self.device)
      )
      object_ang_vel = (
        self.object_ang_vel_w[env_ids]
        if self.object_ang_vel_w is not None
        else torch.zeros(len(env_ids), 3, device=self.device)
      )

      object_state = torch.cat(
        [
          object_pos,
          object_quat,
          object_lin_vel,
          object_ang_vel,
        ],
        dim=-1,
      )

      object.write_root_state_to_sim(object_state, env_ids=env_ids)

    self.robot.write_root_state_to_sim(root_state, env_ids=env_ids)

    self.robot.clear_state(env_ids=env_ids)
    if object is not None:
      object.clear_state(env_ids=env_ids)

    # Reset contact mismatch counter
    self.contact_mismatch_count[env_ids] = 0

  def _update_command(self):
    self.time_steps += 1
    # Check which envs have exceeded their motion's time steps
    env_time_step_totals = self.motion_loader.get_time_step_total(self.motion_indices)
    env_ids = torch.where(self.time_steps >= env_time_step_totals)[0]
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
      # Update each motion FILE's failure distribution separately
      # bin_failed_count and _current_bin_failed are shape (num_motion_files, bin_count)
      # This groups all trajectories from the same file together
      self.bin_failed_count = (
        self.cfg.adaptive_alpha * self._current_bin_failed
        + (1 - self.cfg.adaptive_alpha) * self.bin_failed_count
      )
      self._current_bin_failed.zero_()

    # Update contact indicator from motion data
    if self._has_contact:
      contact_data = self.motion_loader.get_motion_data(
        self.motion_indices, self.time_steps, "object_contact"
      )
      if contact_data.ndim == 1:
        contact_data = contact_data.unsqueeze(1)
      self.ref_object_contact_future = contact_data.unsqueeze(1)
      self.ref_object_contact = contact_data

  def check_contact_mismatch(self, sensor_names: list[str]) -> torch.Tensor:
    """Check if actual contacts match reference contacts.

    Args:
      sensor_names: List of contact sensor names to check

    Returns:
      Boolean tensor of shape (num_envs,) indicating which environments have mismatched contacts
    """
    if self.ref_object_contact is None:
      return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    contact_indicators = self.ref_object_contact  # (num_envs, num_contacts)
    num_matches = min(contact_indicators.shape[1], len(sensor_names))

    # Collect sensor found tensors (minimal Python loop for object access)
    found_tensors = []
    valid_indices = []
    for i in range(num_matches):
      if sensor_names[i] in self._env.scene.sensors:
        found = self._env.scene.sensors[sensor_names[i]].data.found
        if found is not None:
          found_tensors.append(found)
          valid_indices.append(i)

    if not found_tensors:
      return torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)

    # Process each tensor to extract contact flags using torch operations
    has_contact_list = []
    for found in found_tensors:
      has_contact = (found > 0).any(dim=1) if found.dim() > 1 else (found > 0)
      has_contact_list.append(has_contact)

    # Stack into tensor: (num_valid_sensors, num_envs) -> (num_envs, num_valid_sensors)
    has_contact_stacked = (
      torch.stack(has_contact_list, dim=0).transpose(0, 1)
      if len(has_contact_list) > 1
      else has_contact_list[0].unsqueeze(1)
    )

    # Map to full num_matches shape using advanced indexing
    has_contact_all = torch.zeros(
      self.num_envs, num_matches, device=self.device, dtype=torch.bool
    )
    if valid_indices:
      has_contact_all[:, torch.tensor(valid_indices, device=self.device)] = (
        has_contact_stacked
      )

    # Compare reference and actual contacts: mismatch if not all match
    return ~(
      contact_indicators[:, :num_matches] == has_contact_all[:, :num_matches]
    ).all(dim=1)

  def get_per_motion_file_statistics(
    self,
  ) -> dict[str, torch.Tensor | list[str] | list[int]]:
    """Get per-motion-file statistics for detailed analysis.

    Returns:
      Dictionary with per-motion-file tensors:
        - episode_counts: Total episodes per motion file
        - failure_counts: Total early terminations per motion file
        - completion_counts: Total completions per motion file
        - failure_rate: Failure rate per motion file (0-1)
        - completion_rate: Completion rate per motion file (0-1)
        - avg_error: Average anchor position error per motion file
        - bin_failure_distribution: (num_motion_files, bin_count) failure distribution
        - motion_file_names: List of motion file names
        - num_trajectories_per_file: Number of trajectories in each file
    """
    file_avg_error = self._motion_file_error_sum / (
      self._motion_file_error_count + 1e-8
    )
    file_failure_rate = self._motion_file_failure_counts / (
      self._motion_file_episode_counts + 1e-8
    )
    file_completion_rate = self._motion_file_completion_counts / (
      self._motion_file_episode_counts + 1e-8
    )

    return {
      "episode_counts": self._motion_file_episode_counts,
      "failure_counts": self._motion_file_failure_counts,
      "completion_counts": self._motion_file_completion_counts,
      "failure_rate": file_failure_rate,
      "completion_rate": file_completion_rate,
      "avg_error": file_avg_error,
      "bin_failure_distribution": self.bin_failed_count,
      "motion_file_names": self.motion_loader.motion_file_names,
      "num_trajectories_per_file": self.motion_loader.motion_file_trajectory_counts,
    }

  def print_motion_file_summary(self) -> None:
    """Print a summary of per-motion-file performance to console."""
    stats = self.get_per_motion_file_statistics()
    print("\n" + "=" * 70)
    print("Multi-Motion File Performance Summary (Trajectories Grouped)")
    print("=" * 70)
    print(
      f"{'Motion File':<30} {'Trajs':>6} {'Episodes':>10} {'Fail%':>8} {'Comp%':>8} {'AvgErr':>8}"
    )
    print("-" * 70)

    motion_file_names = self.motion_loader.motion_file_names
    num_trajs_per_file = self.motion_loader.motion_file_trajectory_counts
    for i, name in enumerate(motion_file_names):
      episodes = int(stats["episode_counts"][i].item())  # type: ignore[union-attr]
      fail_rate = stats["failure_rate"][i].item() * 100  # type: ignore[union-attr]
      comp_rate = stats["completion_rate"][i].item() * 100  # type: ignore[union-attr]
      avg_err = stats["avg_error"][i].item()  # type: ignore[union-attr]
      n_trajs = num_trajs_per_file[i]
      # Truncate name if too long
      display_name = name[:28] + ".." if len(name) > 30 else name
      print(
        f"{display_name:<30} {n_trajs:>6} {episodes:>10} {fail_rate:>7.1f}% {comp_rate:>7.1f}% {avg_err:>8.3f}"
      )

    print("=" * 70)
    total_episodes = stats["episode_counts"].sum().item()  # type: ignore[union-attr]
    total_trajs = self.motion_loader.num_motions  # Total trajectories across all files
    avg_fail_rate = (
      stats["failure_counts"].sum() / (stats["episode_counts"].sum() + 1e-8)  # type: ignore[union-attr]
    ).item() * 100
    avg_comp_rate = (
      stats["completion_counts"].sum() / (stats["episode_counts"].sum() + 1e-8)  # type: ignore[union-attr]
    ).item() * 100
    print(
      f"{'TOTAL':<30} {total_trajs:>6} {int(total_episodes):>10} {avg_fail_rate:>7.1f}% {avg_comp_rate:>7.1f}%"
    )
    print(f"Number of motion files: {self.num_motion_files}")
    print(f"Total trajectories: {total_trajs}")
    print("=" * 70 + "\n")

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

      object: Entity | None = self._env.scene.entities.get("object")
      if object is not None:
        object_free_joint_q_adr = object.indexing.free_joint_q_adr.cpu().numpy()
        target_pos = self.object_pos_w[visualizer.env_idx].cpu().numpy()
        target_quat = self.object_quat_w[visualizer.env_idx].cpu().numpy()
        qpos[object_free_joint_q_adr[:3]] = target_pos
        qpos[object_free_joint_q_adr[3:7]] = target_quat

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
  adaptive_kernel_size: int = 1
  adaptive_lambda: float = 0.8
  adaptive_uniform_ratio: float = 0.1
  adaptive_alpha: float = 0.001
  sampling_mode: Literal["adaptive", "uniform", "start"] = "adaptive"
  motion_assignment_mode: Literal["random", "linear", "best"] = "random"
  horizon: int = 0
  """Horizon for future motion data. If > 0, properties will return sequences
  of shape (num_envs, horizon, ...) instead of (num_envs, ...). 
  If 0, returns single timestep data as before."""

  @dataclass
  class VizCfg:
    mode: Literal["ghost", "frames"] = "ghost"
    ghost_color: tuple[float, float, float, float] = (0.5, 0.7, 0.5, 0.5)

  viz: VizCfg = field(default_factory=VizCfg)
