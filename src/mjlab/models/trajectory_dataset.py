"""Dataset loader for trajectory data using MultiMotionLoader."""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch.utils.data import Dataset

from mjlab.tasks.tracking.mdp.commands import MultiMotionLoader

if TYPE_CHECKING:
  pass


class TrajectoryDataset(Dataset):
  """Dataset for loading trajectory windows from motion files.

  Uses MultiMotionLoader to load motion data and extracts windows of
  fixed horizon H from the trajectories.
  """

  def __init__(
    self,
    motion_dir: str,
    traj_name_patterns: list[str],
    body_indexes: torch.Tensor,
    anchor_body_index: int,
    horizon: int,
    device: str = "cpu",
    stride: int = 1,
  ):
    """Initialize the trajectory dataset.

    Args:
        motion_dir: Base directory containing trajectory subdirectories
        traj_name_patterns: List of regex patterns to match trajectory names
        body_indexes: Body indices to extract from motion data
        anchor_body_name: Name of the anchor body (e.g., "torso_link")
        body_names: List of body names used in the motion data
        horizon: Number of timesteps in each trajectory window
        device: Device to load tensors on
        stride: Stride for sliding window (1 = every window, 2 = every other window, etc.)
    """
    self.horizon = horizon
    self.device = device
    self.stride = stride

    # Load motion data
    self.motion_loader = MultiMotionLoader(
      motion_dir=motion_dir,
      traj_name_patterns=traj_name_patterns,
      body_indexes=body_indexes,
      device=device,
    )
    self.anchor_body_index = anchor_body_index
    # Extract all valid trajectory windows
    self.windows: list[tuple[int, int]] = []  # List of (motion_idx, start_timestep)

    for motion_idx in range(self.motion_loader.num_motions):
      motion = self.motion_loader.motions[motion_idx]
      time_step_total = motion.time_step_total

      # Extract windows with stride
      for start_t in range(0, time_step_total - horizon + 1, stride):
        self.windows.append((motion_idx, start_t))

    if len(self.windows) == 0:
      raise ValueError(
        f"No valid trajectory windows found. Horizon={horizon}, "
        f"but trajectories have at most {max(m.time_step_total for m in self.motion_loader.motions)} timesteps."
      )

    # Check if motion data is static (all timesteps identical)
    # This is a data quality check
    if len(self.windows) > 0:
      motion0 = self.motion_loader.motions[0]
      if motion0.time_step_total > 1:
        # Check if first two timesteps are identical
        joint_pos_t0 = motion0.joint_pos[0]
        joint_pos_t1 = motion0.joint_pos[1]
        if (joint_pos_t0 == joint_pos_t1).all().item():
          import warnings

          warnings.warn(
            f"WARNING: Motion data appears to be static (all timesteps identical). "
            f"This will result in identical trajectory windows. "
            f"Please check your motion files in {motion_dir}.",
            UserWarning,
            stacklevel=2,
          )

  def __len__(self) -> int:
    return len(self.windows)

  def __getitem__(self, idx: int) -> torch.Tensor:
    """Get a trajectory window.

    Args:
        idx: Index of the window

    Returns:
        Trajectory tensor of shape (horizon, feature_dim) where feature_dim = 72
        Features: [joint_pos (29), joint_vel (29), anchor_pos (3), anchor_quat (4),
                  object_pos (3), object_quat (4)]
    """
    motion_idx, start_t = self.windows[idx]
    motion = self.motion_loader.motions[motion_idx]

    # Extract horizon window
    end_t = start_t + self.horizon

    # Get data for this window
    # Use .clone() to ensure we get a copy, not a view, to avoid any potential issues
    joint_pos = motion.joint_pos[start_t:end_t].clone()  # (horizon, 29)
    joint_vel = motion.joint_vel[start_t:end_t].clone()  # (horizon, 29)

    # Get anchor body pose (from body_pos_w and body_quat_w)
    # body_pos_w already has the selected bodies, so anchor_body_index is the index within body_names
    body_pos_w = motion.body_pos_w[
      start_t:end_t
    ].clone()  # (horizon, num_selected_bodies, 3)
    body_quat_w = motion.body_quat_w[
      start_t:end_t
    ].clone()  # (horizon, num_selected_bodies, 4)
    anchor_pos = body_pos_w[:, self.anchor_body_index]  # (horizon, 3)
    anchor_quat = body_quat_w[:, self.anchor_body_index]  # (horizon, 4)

    # Get object pose
    object_pos = motion.object_pos_w[start_t:end_t].clone()  # (horizon, 3)
    object_quat = motion.object_quat_w[start_t:end_t].clone()  # (horizon, 4)

    # Concatenate all features: (horizon, 72)
    trajectory = torch.cat(
      [
        joint_pos,  # (horizon, 29)
        joint_vel,  # (horizon, 29)
        anchor_pos,  # (horizon, 3)
        anchor_quat,  # (horizon, 4)
        object_pos,  # (horizon, 3)
        object_quat,  # (horizon, 4)
      ],
      dim=1,
    )

    return trajectory

  def get_input_dim(self) -> int:
    """Get the input dimension (72)."""
    return 72
