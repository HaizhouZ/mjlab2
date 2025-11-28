import os
from dataclasses import dataclass
from typing import Any

import mujoco
import numpy as np
import torch
import tyro
from tqdm import tqdm

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
  unitree_g1_flat_tracking_env_cfg_box,
)
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
)


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
  return q / torch.norm(q, dim=-1, keepdim=True).clamp_min(1e-8)


def _quat_to_yaw(q: torch.Tensor) -> torch.Tensor:
  """Return yaw (rotation about Z) for quaternions shaped (..., 4)."""
  if q.shape[-1] != 4:
    raise ValueError(f"Quaternion must have last dim 4, got {q.shape}")
  w, x, y, z = torch.unbind(q, dim=-1)
  siny_cosp = 2.0 * (w * z + x * y)
  cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
  return torch.atan2(siny_cosp, cosy_cosp)


def _yaw_to_quat(yaw: torch.Tensor) -> torch.Tensor:
  """Create yaw-only quaternion (w, x, y, z) from yaw angles (radians)."""
  yaw = yaw.to(torch.float32)
  half = 0.5 * yaw
  sin_half = torch.sin(half)
  cos_half = torch.cos(half)
  zeros = torch.zeros_like(half)
  return torch.stack((cos_half, zeros, zeros, sin_half), dim=-1)


def _extract_last_frame_yaw(input_file: str, device: torch.device | str) -> torch.Tensor:
  """Load input motion and return yaw of the final frame."""
  with np.load(input_file, allow_pickle=True) as data:
    if "base_xyz_quat" in data:
      base_quats = data["base_xyz_quat"].astype(np.float32)[:, 3:7]
    elif "x" in data:
      x_np = data["x"].astype(np.float32)
      if x_np.ndim != 2 or x_np.shape[1] < 7:
        raise ValueError(
          f"'x' array must be 2D with >=7 columns to extract base quat, got {x_np.shape}"
        )
      base_quats = x_np[:, 3:7]
    else:
      raise ValueError(
        f"Cannot determine base orientation from {input_file}: missing 'base_xyz_quat' or 'x'"
      )

  if base_quats.shape[0] == 0:
    raise ValueError(f"No frames found in {input_file}")

  last_quat = torch.from_numpy(base_quats[-1:]).to(device)
  last_quat = _normalize_quat(last_quat)
  return _quat_to_yaw(last_quat)[0]


def _extract_first_frame_base_pos(
  input_file: str, device: torch.device | str
) -> torch.Tensor:
  """Load input motion and return xyz position of the first frame."""
  with np.load(input_file, allow_pickle=True) as data:
    if "base_xyz_quat" in data:
      base_pos = data["base_xyz_quat"].astype(np.float32)[:, 0:3]
    elif "x" in data:
      x_np = data["x"].astype(np.float32)
      if x_np.ndim != 2 or x_np.shape[1] < 3:
        raise ValueError(
          f"'x' array must be 2D with >=3 columns to extract base position, got {x_np.shape}"
        )
      base_pos = x_np[:, 0:3]
    else:
      raise ValueError(
        f"Cannot determine base position from {input_file}: missing 'base_xyz_quat' or 'x'"
      )

  if base_pos.shape[0] == 0:
    raise ValueError(f"No frames found in {input_file}")

  first_pos = torch.from_numpy(base_pos[0:1]).to(device)
  return first_pos[0]


def _extract_first_frame_base_rot(
  input_file: str, device: torch.device | str
) -> torch.Tensor:
  """Load input motion and return quaternion of the first frame."""
  with np.load(input_file, allow_pickle=True) as data:
    if "base_xyz_quat" in data:
      base_quats = data["base_xyz_quat"].astype(np.float32)[:, 3:7]
    elif "x" in data:
      x_np = data["x"].astype(np.float32)
      if x_np.ndim != 2 or x_np.shape[1] < 7:
        raise ValueError(
          f"'x' array must be 2D with >=7 columns to extract base quat, got {x_np.shape}"
        )
      base_quats = x_np[:, 3:7]
    else:
      raise ValueError(
        f"Cannot determine base orientation from {input_file}: missing 'base_xyz_quat' or 'x'"
      )

  if base_quats.shape[0] == 0:
    raise ValueError(f"No frames found in {input_file}")

  first_quat = torch.from_numpy(base_quats[0:1]).to(device)
  first_quat = _normalize_quat(first_quat)
  return first_quat[0]


def extract_contacts_from_distance(
  eef_positions: np.ndarray,
  object_position: np.ndarray,
  object_rotation: np.ndarray | None = None,
  object_size: np.ndarray | None = None,
  threshold: float = 0.05,
) -> dict[str, Any]:
  """Extract contact information using distance threshold with object size consideration.

  This is a simpler fallback method that checks if end effector positions are within
  the object's bounding box (accounting for size) plus a threshold margin.

  Args:
    eef_positions: (num_eefs, 3) array of end effector positions in world frame
    object_position: (3,) array of object position (center) in world frame
    object_rotation: (4,) array of object quaternion (w, x, y, z) in world frame, or None for axis-aligned
    object_size: (3,) array of object half-extents (size in MuJoCo convention), or None to use threshold only
    threshold: Distance threshold in meters for contact detection (default: 0.05m = 5cm)
               If object_size is None, this is used as distance from center.
               If object_size is provided, this is added as margin around the bounding box.

  Returns:
    Dictionary with contact information:
      - contact_positions: (num_eefs, 3) array of contact positions (closest point on object if in contact, NaN otherwise)
      - contact_indicators: (num_eefs,) boolean array indicating if contact exists
  """
  num_eefs = eef_positions.shape[0]
  contact_indicators = np.zeros(num_eefs, dtype=bool)
  contact_positions = np.full((num_eefs, 3), np.nan, dtype=np.float32)

  # If object size is not provided, use simple distance from center
  if object_size is None:
    for i in range(num_eefs):
      dist = np.linalg.norm(eef_positions[i] - object_position)
      if dist <= threshold:
        contact_indicators[i] = True
        contact_positions[i] = object_position.copy()
  else:
    # Transform end effector positions to object's local frame
    if object_rotation is not None:
      # Convert quaternion to rotation matrix and transform
      # quat_apply_inverse: transform from world to object frame
      # Ensure float32 dtype for torch operations
      eef_pos_local = quat_apply_inverse(
        torch.from_numpy(object_rotation.astype(np.float32)).unsqueeze(0),
        torch.from_numpy(
          (eef_positions - object_position.reshape(1, 3)).astype(np.float32)
        ),
      ).numpy()
    else:
      # No rotation, just translate
      eef_pos_local = eef_positions - object_position.reshape(1, 3)

    # Check if end effector is within bounding box + threshold
    # object_size is half-extents, so bounds are [-size-threshold, +size+threshold]
    for i in range(num_eefs):
      # Skip if position is NaN (end effector not found)
      if np.isnan(eef_pos_local[i]).any():
        continue

      # Check if within bounding box (with threshold margin)
      # object_size + threshold broadcasts correctly: (3,) + scalar -> (3,)
      bounds = object_size + threshold
      within_bounds = np.all(np.abs(eef_pos_local[i]) <= bounds)

      if within_bounds:
        contact_indicators[i] = True
        # Find closest point on the cube surface
        # Clamp to bounding box bounds
        closest_local = np.clip(eef_pos_local[i], -object_size, object_size)
        # Transform back to world frame
        if object_rotation is not None:
          # Ensure float32 dtype for torch operations
          closest_world = (
            quat_apply(
              torch.from_numpy(object_rotation.astype(np.float32)).unsqueeze(0),
              torch.from_numpy(closest_local.astype(np.float32)).unsqueeze(0),
            ).numpy()[0]
            + object_position
          )
        else:
          closest_world = closest_local + object_position
        contact_positions[i] = closest_world

  return {
    "contact_positions": contact_positions,
    "contact_indicators": contact_indicators,
  }


@dataclass
class StartInterpolationConfig:
  frames: int
  base_pos: torch.Tensor
  base_rot: torch.Tensor
  joint_pos: torch.Tensor
  easing: str = "smoothstep"


@dataclass
class StandupInterpolationConfig:
  frames: int
  target_base_pos: torch.Tensor
  target_base_rot: torch.Tensor
  target_joint_pos: torch.Tensor
  target_base_height: float | None = None
  preserve_xy: bool = True
  easing: str = "smoothstep"


class TrajectoryNpzSimLoader:
  def __init__(
    self,
    input_file: str,
    speed: float,
    device: torch.device | str,
    repeat_last_frame: int = 0,
    repeat_first_frame: int = 0,
    append_reverse: bool = False,
    start_cfg: StartInterpolationConfig | None = None,
    standup_cfg: StandupInterpolationConfig | None = None,
  ):
    self.input_file = input_file
    self.speed = float(speed)
    self.device = device
    self.current_idx = 0
    self.repeat_last_frame = max(0, int(repeat_last_frame))
    self.repeat_first_frame = max(0, int(repeat_first_frame))
    self.append_reverse = bool(append_reverse)
    self.start_cfg = start_cfg
    self.standup_cfg = standup_cfg

    self._load()
    # Calculate output_fps from input_fps and speed
    self.output_fps = int(round(self.input_fps / self.speed))
    self.output_dt = 1.0 / float(self.output_fps)

    print("[INFO]: Input FPS: ", self.input_fps)
    print("[INFO]: Speed: ", self.speed)
    print("[INFO]: Output FPS: ", self.output_fps)
    print("[INFO]: Output DT: ", self.output_dt)

    self._resample_to_output_fps_using_times()
    self._prepend_start_transition()
    self._repeat_first_frame()
    self._append_standup_transition()
    self._repeat_last_frame()
    # Compute velocities before appending reverse so we can properly reverse and negate them
    self._compute_velocities_from_resampled()
    self._append_reverse()

  def _load_ilyass_pickle(self, file_path: str):
    """Load Ilyass-format pickle containing:
    fps, root_pos, root_rot(xyzw), dof_pos, object_pos, object_rot(xyzw)
    """

    import pickle

    with open(file_path, "rb") as f:
      data = pickle.load(f)

    required_keys = [
      "fps",
      "root_pos",
      "root_rot",
      "dof_pos",
      "object_pos",
      "object_rot",
    ]
    for k in required_keys:
      if k not in data:
        raise ValueError(f"Missing key '{k}' in Ilyass pickle")

    # Extract
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)  # (T, 3)
    root_rot_xyzw = np.asarray(data["root_rot"], dtype=np.float32)  # (T, 4)
    dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)  # (T, 29)
    obj_pos = np.asarray(data["object_pos"], dtype=np.float32)  # (T, 3)
    obj_rot_xyzw = np.asarray(data["object_rot"], dtype=np.float32)  # (T, 4)
    fps = int(data["fps"])

    T = root_pos.shape[0]

    # Build artificial time array
    times_np = np.arange(T, dtype=np.float32) * (1.0 / fps)

    # Convert xyzw → wxyz
    root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]
    obj_rot_wxyz = obj_rot_xyzw[:, [3, 0, 1, 2]]

    # Store internally in the same format as the Standard NPZ loader
    self.input_times_np = times_np
    self.input_times = torch.from_numpy(times_np).to(self.device)
    self.input_frames = T

    self.motion_base_poss_input = torch.from_numpy(root_pos).to(self.device)
    self.motion_base_rots_input = _normalize_quat(
      torch.from_numpy(root_rot_wxyz).to(self.device)
    )
    self.motion_dof_poss_input = torch.from_numpy(dof_pos).to(self.device)

    # Object is always present in this format
    self.has_object = True
    self.object_pos_input = torch.from_numpy(obj_pos).to(self.device)
    self.object_rots_input = _normalize_quat(
      torch.from_numpy(obj_rot_wxyz).to(self.device)
    )

    # Timing info
    self.duration = float(times_np[-1] - times_np[0]) if T > 1 else 0.0
    self.input_dt = float(self.duration / max(1, T - 1)) if T > 1 else 1.0 / fps
    self.input_fps = fps

  def _load(self) -> None:
    """Unified loader for Victor NPZ, Standard NPZ, and Ilyass PKL formats."""

    # Automatically detect pickle format
    if self.input_file.endswith(".pkl"):
      print("[Loader] Detected Ilyass pickle format.")
      return self._load_ilyass_pickle(self.input_file)

    # Otherwise load NPZ
    data = np.load(self.input_file, allow_pickle=True)
    print("[Loader] Detected NPZ format.")

    # Victor format
    is_victor_format = "base_xyz_quat" in data and "actuator_pos" in data

    if is_victor_format:
      print("[Loader] Using Victor format loader.")
      times_np = data["time"].astype(np.float32)

      base_xyz_quat = data["base_xyz_quat"].astype(np.float32)
      actuator_pos = data["actuator_pos"].astype(np.float32)

      T = times_np.shape[0]
      self.input_times_np = times_np
      self.input_times = torch.from_numpy(times_np).to(self.device)

      self.motion_base_poss_input = torch.from_numpy(base_xyz_quat[:, 0:3]).to(
        self.device
      )
      self.motion_base_rots_input = _normalize_quat(
        torch.from_numpy(base_xyz_quat[:, 3:7]).to(self.device)
      )
      self.motion_dof_poss_input = torch.from_numpy(actuator_pos).to(self.device)

      self.has_object = False
      if "obj_0_xyz_quat" in data:
        obj = data["obj_0_xyz_quat"].astype(np.float32)
        self.has_object = True
        self.object_pos_input = torch.from_numpy(obj[:, 0:3]).to(self.device)
        self.object_rots_input = _normalize_quat(
          torch.from_numpy(obj[:, 3:7]).to(self.device)
        )

    else:
      print("[Loader] Using Standard NPZ format loader.")
      assert "x" in data, f"Key 'x' not found in standard NPZ file: {self.input_file}"
      x_np = data["x"].astype(np.float32)

      times_np = data["time"].astype(np.float32)
      T = x_np.shape[0]

      self.input_times_np = times_np
      self.input_times = torch.from_numpy(times_np).to(self.device)

      rs = torch.from_numpy(x_np).to(self.device)

      self.motion_base_poss_input = rs[:, 0:3]
      self.motion_base_rots_input = _normalize_quat(rs[:, 3:7])
      self.motion_dof_poss_input = rs[:, 7:36]

      # Optional embedded object: identical to your previous code
      self.has_object = False
      if x_np.shape[1] >= 43:  # qpos(43) → includes object (quat + pos)
        self.has_object = True
        self.object_rots_input = _normalize_quat(rs[:, 36:40])
        self.object_pos_input = rs[:, 40:43]

    # Timing setup for all formats
    self.input_frames = T
    self.duration = float(times_np[-1] - times_np[0]) if T > 1 else 0.0
    self.input_dt = float(self.duration / max(1, T - 1)) if T > 1 else self.output_dt
    self.input_fps = int(round(1.0 / max(1e-8, self.input_dt)))

  def _resample_to_output_fps_using_times(self) -> None:
    if self.input_frames <= 1:
      self.motion_base_poss = self.motion_base_poss_input
      self.motion_base_rots = self.motion_base_rots_input
      self.motion_dof_poss = self.motion_dof_poss_input
      if self.has_object:
        self.object_poss = self.object_pos_input
        self.object_rots = self.object_rots_input
      self.output_frames = self.input_frames
      return

    t0 = float(self.input_times_np[0])
    duration = float(self.input_times_np[-1] - self.input_times_np[0])
    if duration <= 0.0:
      self.motion_base_poss = self.motion_base_poss_input
      self.motion_base_rots = self.motion_base_rots_input
      self.motion_dof_poss = self.motion_dof_poss_input
      if self.has_object:
        self.object_poss = self.object_pos_input
        self.object_rots = self.object_rots_input
      self.output_frames = self.input_frames
      return

    times_out = np.arange(0.0, duration + 1e-8, self.output_dt, dtype=np.float32) + t0
    self.output_frames = int(times_out.shape[0])

    idx1 = np.searchsorted(self.input_times_np, times_out, side="right")
    idx1 = np.clip(idx1, 1, self.input_frames - 1)
    idx0 = idx1 - 1
    t0s = self.input_times_np[idx0]
    t1s = self.input_times_np[idx1]
    denom = np.maximum(1e-8, (t1s - t0s))
    blend_np = (times_out - t0s) / denom

    index_0 = torch.from_numpy(idx0.astype(np.int64)).to(self.device)
    index_1 = torch.from_numpy(idx1.astype(np.int64)).to(self.device)
    blend = torch.from_numpy(blend_np.astype(np.float32)).to(self.device)

    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    q0 = self.motion_base_rots_input[index_0]
    q1 = self.motion_base_rots_input[index_1]
    dot = (q0 * q1).sum(-1, keepdim=True)
    q1 = torch.where(dot < 0, -q1, q1)
    q = q0 * (1 - blend.unsqueeze(1)) + q1 * blend.unsqueeze(1)
    self.motion_base_rots = _normalize_quat(q)
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )

    if self.has_object:
      self.object_poss = self._lerp(
        self.object_pos_input[index_0],
        self.object_pos_input[index_1],
        blend.unsqueeze(1),
      )
      oq0 = self.object_rots_input[index_0]
      oq1 = self.object_rots_input[index_1]
      odot = (oq0 * oq1).sum(-1, keepdim=True)
      oq1 = torch.where(odot < 0, -oq1, oq1)
      oq = oq0 * (1 - blend.unsqueeze(1)) + oq1 * blend.unsqueeze(1)
      self.object_rots = _normalize_quat(oq)

  def _append_standup_transition(self) -> None:
    """Append a transition to the nominal stand pose after the motion tail."""
    cfg = getattr(self, "standup_cfg", None)
    if cfg is None or cfg.frames <= 0:
      return

    total_frames = int(self.motion_base_poss.shape[0])
    if total_frames <= 0:
      return

    frames = int(cfg.frames)
    blend = torch.linspace(0.0, 1.0, frames + 1, device=self.device)[1:].unsqueeze(1)
    easing = (cfg.easing or "").lower()
    if easing in ("smoothstep", "cubic"):
      blend = blend * blend * (3.0 - 2.0 * blend)
    elif easing in ("cosine", "cos"):
      blend = 0.5 - 0.5 * torch.cos(blend * torch.pi)
    # Guarantee the final frame reaches the target exactly.
    blend[-1] = 1.0

    target_base_pos = cfg.target_base_pos.to(self.device).clone()
    last_base_pos = self.motion_base_poss[-1].clone()
    if cfg.preserve_xy:
      target_base_pos[:2] = last_base_pos[:2]
    if cfg.target_base_height is not None:
      target_base_pos[2] = float(cfg.target_base_height)

    target_pos = target_base_pos.unsqueeze(0).expand(frames, -1)
    start_pos = last_base_pos.unsqueeze(0).expand(frames, -1)
    appended_pos = self._lerp(start_pos, target_pos, blend)

    target_base_rot = _normalize_quat(cfg.target_base_rot.to(self.device).unsqueeze(0))
    target_base_rot = target_base_rot.expand(frames, -1)
    base_rot_segment = self.motion_base_rots[-1].unsqueeze(0).expand(frames, -1)
    dot = (base_rot_segment * target_base_rot).sum(-1, keepdim=True)
    target_base_rot = torch.where(dot < 0, -target_base_rot, target_base_rot)
    blended_rot = base_rot_segment * (1.0 - blend) + target_base_rot * blend
    appended_rot = _normalize_quat(blended_rot)

    target_joint_pos = cfg.target_joint_pos.to(self.device).unsqueeze(0).expand(frames, -1)
    start_joint = self.motion_dof_poss[-1].unsqueeze(0).expand(frames, -1)
    appended_joint = self._lerp(start_joint, target_joint_pos, blend)

    self.motion_base_poss = torch.cat([self.motion_base_poss, appended_pos], dim=0)
    self.motion_base_rots = torch.cat([self.motion_base_rots, appended_rot], dim=0)
    self.motion_dof_poss = torch.cat([self.motion_dof_poss, appended_joint], dim=0)

    if self.has_object:
      last_obj_pos = self.object_poss[-1:].repeat(frames, 1)
      last_obj_rot = self.object_rots[-1:].repeat(frames, 1)
      self.object_poss = torch.cat([self.object_poss, last_obj_pos], dim=0)
      self.object_rots = torch.cat([self.object_rots, last_obj_rot], dim=0)

    self.output_frames += frames

  def _prepend_start_transition(self) -> None:
    """Prepend frames interpolating from a nominal start pose to the first motion frame."""
    cfg = getattr(self, "start_cfg", None)
    if cfg is None or cfg.frames <= 0:
      return

    if self.motion_base_poss.shape[0] <= 0:
      return

    frames = int(cfg.frames)
    blend = torch.linspace(0.0, 1.0, frames + 1, device=self.device)[:-1].unsqueeze(1)
    easing = (cfg.easing or "").lower()
    if easing in ("smoothstep", "cubic"):
      blend = blend * blend * (3.0 - 2.0 * blend)
    elif easing in ("cosine", "cos"):
      blend = 0.5 - 0.5 * torch.cos(blend * torch.pi)

    start_base_pos = cfg.base_pos.to(self.device).clone()
    # Use the exact first frame position from the motion data (after resampling).
    target_base_pos = self.motion_base_poss[0].clone()
    start_pos = start_base_pos.unsqueeze(0).expand(frames, -1)
    target_pos = target_base_pos.unsqueeze(0).expand(frames, -1)
    prepended_pos = self._lerp(start_pos, target_pos, blend)

    # Interpolate rotation: start from yaw-only (upright) and blend to full target rotation.
    # This preserves the final pitch/roll from the motion data to avoid discontinuity.
    start_base_rot = _normalize_quat(cfg.base_rot.to(self.device).unsqueeze(0))
    # Use the FULL target rotation (including pitch/roll) to avoid discontinuity at boundary
    target_base_rot = _normalize_quat(self.motion_base_rots[0].unsqueeze(0))
    # Ensure quaternions are in the same hemisphere for proper interpolation
    dot = (start_base_rot * target_base_rot).sum(-1, keepdim=True)
    target_base_rot = torch.where(dot < 0, -target_base_rot, target_base_rot)
    # Expand to all frames for interpolation
    start_rot_expanded = start_base_rot.expand(frames, -1)
    target_rot_expanded = target_base_rot.expand(frames, -1)
    # Spherical linear interpolation (SLERP) approximated by normalized lerp
    blended_rot = start_rot_expanded * (1.0 - blend) + target_rot_expanded * blend
    prepended_rot = _normalize_quat(blended_rot)

    start_joint_pos = cfg.joint_pos.to(self.device).unsqueeze(0).expand(frames, -1)
    target_joint_pos = self.motion_dof_poss[0].unsqueeze(0).expand(frames, -1)
    prepended_joint = self._lerp(start_joint_pos, target_joint_pos, blend)

    self.motion_base_poss = torch.cat([prepended_pos, self.motion_base_poss], dim=0)
    self.motion_base_rots = torch.cat([prepended_rot, self.motion_base_rots], dim=0)
    self.motion_dof_poss = torch.cat([prepended_joint, self.motion_dof_poss], dim=0)

    if self.has_object:
      first_obj_pos = self.object_poss[:1].repeat(frames, 1)
      first_obj_rot = self.object_rots[:1].repeat(frames, 1)
      self.object_poss = torch.cat([first_obj_pos, self.object_poss], dim=0)
      self.object_rots = torch.cat([first_obj_rot, self.object_rots], dim=0)

    self.output_frames += frames

  def _repeat_first_frame(self) -> None:
    """Prepend the first frame N times to extend the trajectory."""
    if self.repeat_first_frame <= 0:
      return

    first_pos = self.motion_base_poss[:1].repeat(self.repeat_first_frame, 1)
    first_rot = self.motion_base_rots[:1].repeat(self.repeat_first_frame, 1)
    first_dof = self.motion_dof_poss[:1].repeat(self.repeat_first_frame, 1)

    self.motion_base_poss = torch.cat([first_pos, self.motion_base_poss], dim=0)
    self.motion_base_rots = torch.cat([first_rot, self.motion_base_rots], dim=0)
    self.motion_dof_poss = torch.cat([first_dof, self.motion_dof_poss], dim=0)

    if self.has_object:
      first_obj_pos = self.object_poss[:1].repeat(self.repeat_first_frame, 1)
      first_obj_rot = self.object_rots[:1].repeat(self.repeat_first_frame, 1)
      self.object_poss = torch.cat([first_obj_pos, self.object_poss], dim=0)
      self.object_rots = torch.cat([first_obj_rot, self.object_rots], dim=0)

    self.output_frames += self.repeat_first_frame

  def _repeat_last_frame(self) -> None:
    """Append the last frame N times to extend the trajectory."""
    if self.repeat_last_frame <= 0:
      return

    last_pos = self.motion_base_poss[-1:].repeat(self.repeat_last_frame, 1)
    last_rot = self.motion_base_rots[-1:].repeat(self.repeat_last_frame, 1)
    last_dof = self.motion_dof_poss[-1:].repeat(self.repeat_last_frame, 1)

    self.motion_base_poss = torch.cat([self.motion_base_poss, last_pos], dim=0)
    self.motion_base_rots = torch.cat([self.motion_base_rots, last_rot], dim=0)
    self.motion_dof_poss = torch.cat([self.motion_dof_poss, last_dof], dim=0)

    if self.has_object:
      last_obj_pos = self.object_poss[-1:].repeat(self.repeat_last_frame, 1)
      last_obj_rot = self.object_rots[-1:].repeat(self.repeat_last_frame, 1)
      self.object_poss = torch.cat([self.object_poss, last_obj_pos], dim=0)
      self.object_rots = torch.cat([self.object_rots, last_obj_rot], dim=0)

    self.output_frames += self.repeat_last_frame

  def _append_reverse(self) -> None:
    """Append the reversed motion trajectory at the end.

    Reverses positions, rotations, and velocities. Velocities are negated
    because the motion is going backwards. The last frame of the original
    trajectory becomes the first frame of the reversed trajectory (duplicate),
    which creates a smooth transition.
    """
    if not self.append_reverse:
      return
    # Store original frame count before appending
    original_frames = self.output_frames

    # Reverse all motion data along the time dimension (dim=0)
    reversed_pos = torch.flip(self.motion_base_poss, dims=[0])
    reversed_rot = torch.flip(self.motion_base_rots, dims=[0])
    reversed_dof = torch.flip(self.motion_dof_poss, dims=[0])

    # Reverse and negate velocities (velocities should point in opposite direction)
    reversed_lin_vel = -torch.flip(self.motion_base_lin_vels, dims=[0])
    reversed_ang_vel = -torch.flip(self.motion_base_ang_vels, dims=[0])
    reversed_dof_vel = -torch.flip(self.motion_dof_vels, dims=[0])

    self.motion_base_poss = torch.cat([self.motion_base_poss, reversed_pos], dim=0)
    self.motion_base_rots = torch.cat([self.motion_base_rots, reversed_rot], dim=0)
    self.motion_dof_poss = torch.cat([self.motion_dof_poss, reversed_dof], dim=0)

    # Append reversed and negated velocities
    self.motion_base_lin_vels = torch.cat(
      [self.motion_base_lin_vels, reversed_lin_vel], dim=0
    )
    self.motion_base_ang_vels = torch.cat(
      [self.motion_base_ang_vels, reversed_ang_vel], dim=0
    )
    self.motion_dof_vels = torch.cat([self.motion_dof_vels, reversed_dof_vel], dim=0)

    if self.has_object:
      reversed_obj_pos = torch.flip(self.object_poss, dims=[0])
      reversed_obj_rot = torch.flip(self.object_rots, dims=[0])
      self.object_poss = torch.cat([self.object_poss, reversed_obj_pos], dim=0)
      self.object_rots = torch.cat([self.object_rots, reversed_obj_rot], dim=0)

    self.output_frames += original_frames

  def _compute_velocities_from_resampled(self) -> None:
    # lin vel from pos derivative and ang vel from quaternion finite difference
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]

    q = self.motion_base_rots
    if q.shape[0] >= 3:
      q_prev, q_next = q[:-2], q[2:]
      q_rel = quat_mul(q_next, quat_conjugate(q_prev))
      omega = axis_angle_from_quat(q_rel) / (2.0 * self.output_dt)
      omega = torch.cat([omega[:1], omega, omega[-1:]], dim=0)
    else:
      omega = torch.zeros_like(self.motion_base_poss)
    self.motion_base_ang_vels = omega

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    return a * (1 - blend) + b * blend

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ],
    bool,
  ]:
    state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= getattr(self, "output_frames", self.input_frames):
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def convert(
  input_file: str,
  output_dir: str,
  output_name: str,
  speed: float = 1.0,
  device: str = "cuda:0",
  repeat_last_frame: int = 0,
  repeat_first_frame: int = 0,
  start_frames: int = 0,
  start_easing: str = "smoothstep",
  append_reverse: bool = False,
  extract_object_states: bool = True,
  contact_method: str = "distance",
  contact_threshold: float = 0.05,
  standup_frames: int = 0,
  standup_base_height: float | None = None,
  standup_preserve_xy: bool = True,
  standup_easing: str = "smoothstep",
  object_position_offset: tuple[float, float, float] | None = None,
  object_rotation_offset: tuple[float, float, float, float] | None = None,
  align_first_frame: bool = False,
):
  """Convert dataset NPZ (qpos in x) to mjlab motion.npz using simulation states.

  Extracts data for all robot bodies to match the expected mjlab motion format.
  Supports both Victor format (separate arrays) and standard format (x array).

  When object data is present, also extracts contact information for end effectors.

  Args:
    input_file: Path to input NPZ file
    output_dir: Base output directory
    output_name: Name for the output (used to create motions/output/<output_name>/motion.npz)
    speed: Speed multiplier for trajectory (output_fps = input_fps * speed)
    device: Device to use for simulation
    repeat_last_frame: Number of times to repeat the last frame
    repeat_first_frame: Number of times to repeat the first frame
    start_frames: Number of frames to interpolate from default stand pose to the first motion frame
    start_easing: Easing function for the start interpolation ("smoothstep", "cosine", "linear")
    append_reverse: If True, append the reversed motion trajectory at the end
    extract_object_states: If False, skip extracting object states and contact information
    contact_method: Method to extract contacts - "mujoco" (preferred, uses physics) or "distance" (uses threshold)
    contact_threshold: Distance threshold in meters for "distance" method (default: 0.05m = 5cm)
    standup_frames: Number of tail frames to blend to the nominal stand pose (0 disables)
    standup_base_height: Optional absolute Z height for the final base pose (None keeps nominal height)
    standup_preserve_xy: Keep the final XY position of the motion when blending to the stand pose
    standup_easing: Easing function for the blend ("smoothstep", "cosine", "linear")
    object_position_offset: Optional constant XYZ offset (meters) applied to the tracked object's pose
    object_rotation_offset: Optional constant quaternion (w, x, y, z) applied to the tracked object's pose
    align_first_frame: Translate XY and rotate yaw so the first frame aligns with world axes
  """
  # Construct output path: motions/output/<output_name>/motion.npz
  output_path = f"motions/output/{output_name}/motion.npz"
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  output_file = output_path

  motion = TrajectoryNpzSimLoader(
    input_file=input_file,
    speed=speed,
    device=device,
    repeat_last_frame=repeat_last_frame,
    repeat_first_frame=repeat_first_frame,
    append_reverse=append_reverse,
  )

  # Get output_fps from motion loader (calculated from input_fps * speed)
  output_fps = motion.output_fps

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / float(output_fps)
  if extract_object_states:
    scene = Scene(unitree_g1_flat_tracking_env_cfg_box().scene, device=device)
  else:
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot: Entity = scene["robot"]
  box: Entity | None = scene.entities.get("box") if hasattr(scene, "entities") else None
  obj_pos_offset = None
  if object_position_offset is not None:
    obj_pos_offset = torch.tensor(
      object_position_offset, dtype=torch.float32, device=sim.device
    )

  obj_rot_offset = None
  if object_rotation_offset is not None:
    obj_rot_offset = _normalize_quat(
      torch.tensor(object_rotation_offset, dtype=torch.float32, device=sim.device)
      .unsqueeze(0)
      .clone()
    )[0]


  default_root_state = robot.data.default_root_state[0].detach().clone().to(sim.device)
  default_base_pos = default_root_state[0:3].clone()
  default_base_rot = _normalize_quat(default_root_state[3:7].clone().unsqueeze(0))[0]
  default_joint_pos = robot.data.default_joint_pos[0].detach().clone().to(sim.device)

  start_cfg: StartInterpolationConfig | None = None
  if start_frames > 0:
    first_frame_base_pos = _extract_first_frame_base_pos(input_file, sim.device)
    first_frame_base_rot = _extract_first_frame_base_rot(input_file, sim.device)
    first_frame_yaw = _quat_to_yaw(first_frame_base_rot.unsqueeze(0))[0]
    first_frame_yaw_quat = _normalize_quat(
      _yaw_to_quat(first_frame_yaw.unsqueeze(0))
    )[0]
    start_base_pos = default_base_pos.clone()
    start_base_pos[:2] = first_frame_base_pos[:2]
    start_cfg = StartInterpolationConfig(
      frames=start_frames,
      base_pos=start_base_pos,
      base_rot=first_frame_yaw_quat.clone(),
      joint_pos=default_joint_pos.clone(),
      easing=start_easing,
    )

  standup_cfg: StandupInterpolationConfig | None = None
  if standup_frames > 0:
    base_pos = default_base_pos.clone()

    last_frame_yaw = _extract_last_frame_yaw(input_file, sim.device)
    base_rot = _normalize_quat(_yaw_to_quat(last_frame_yaw).unsqueeze(0))[0]
    target_joint_pos = default_joint_pos.clone()
    standup_cfg = StandupInterpolationConfig(
      frames=standup_frames,
      target_base_pos=base_pos,
      target_base_rot=base_rot,
      target_joint_pos=target_joint_pos,
      target_base_height=standup_base_height,
      preserve_xy=standup_preserve_xy,
      easing=standup_easing,
    )

  motion = TrajectoryNpzSimLoader(
    input_file=input_file,
    output_fps=int(round(output_fps)),
    device=sim.device,
    repeat_last_frame=repeat_last_frame,
    repeat_first_frame=repeat_first_frame,
    append_reverse=append_reverse,
    start_cfg=start_cfg,
    standup_cfg=standup_cfg,
  )

  total_frames = getattr(motion, "output_frames", motion.input_frames)
  if align_first_frame and total_frames > 0:
    first_pos = motion.motion_base_poss[0].clone()
    xy_offset = first_pos[:2].clone()
    if xy_offset.abs().sum() > 0:
      motion.motion_base_poss[:, :2] -= xy_offset
      if hasattr(motion, "motion_base_poss_input"):
        motion.motion_base_poss_input[:, :2] -= xy_offset
      if getattr(motion, "has_object", False):
        motion.object_poss[:, :2] -= xy_offset
        if hasattr(motion, "object_pos_input"):
          motion.object_pos_input[:, :2] -= xy_offset

    first_rot = motion.motion_base_rots[0:1]
    first_yaw = _quat_to_yaw(first_rot)[0]
    delta_yaw = -first_yaw
    delta_quat = _normalize_quat(_yaw_to_quat(delta_yaw.unsqueeze(0)))[0]
    delta_quat_full = delta_quat.unsqueeze(0)

    rot_expand = delta_quat_full.expand_as(motion.motion_base_rots)
    motion.motion_base_rots = _normalize_quat(
      quat_mul(rot_expand, motion.motion_base_rots)
    )

    vec_expand = delta_quat_full.expand(motion.motion_base_poss.shape[0], -1)
    motion.motion_base_poss = quat_apply(vec_expand, motion.motion_base_poss)
    motion.motion_base_lin_vels = quat_apply(vec_expand, motion.motion_base_lin_vels)
    motion.motion_base_ang_vels = quat_apply(vec_expand, motion.motion_base_ang_vels)

    if getattr(motion, "has_object", False):
      obj_vec_expand = delta_quat_full.expand(motion.object_poss.shape[0], -1)
      obj_rot_expand = delta_quat_full.expand(motion.object_rots.shape[0], -1)
      motion.object_poss = quat_apply(obj_vec_expand, motion.object_poss)
      motion.object_rots = _normalize_quat(
        quat_mul(obj_rot_expand, motion.object_rots)
      )

      if hasattr(motion, "object_pos_input"):
        motion.object_pos_input = quat_apply(
          delta_quat_full.expand(motion.object_pos_input.shape[0], -1),
          motion.object_pos_input,
        )
      if hasattr(motion, "object_rots_input"):
        motion.object_rots_input = _normalize_quat(
          quat_mul(
            delta_quat_full.expand(motion.object_rots_input.shape[0], -1),
            motion.object_rots_input,
          )
        )


  log_object = (
    extract_object_states
    and getattr(motion, "has_object", False)
    and (box is not None and not box.data.is_fixed_base)
  )

  # Extract ALL bodies - mjlab expects complete body data for all robot bodies
  # The body_names in the config are used for tracking/observation, but motion files need all bodies
  print(f"Robot has {len(robot.body_names)} bodies: {robot.body_names}")

  # End effector site names for contact detection
  # eef_names = ["left_palm", "right_palm"]
  # # Get end effector site indices in robot.site_names
  # eef_indices, eef_names_found = robot.find_sites(eef_names, preserve_order=True)
  # for i, eef_name in enumerate(eef_names):
  #   if i >= len(eef_indices) or eef_indices[i] < 0:
  #     print(f"Warning: End effector site '{eef_name}' not found in robot site names")
  #     if i >= len(eef_indices):
  #       eef_indices.append(-1)
  #     else:
  #       eef_indices[i] = -1

  eef_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
  eef_indices, eef_names_found = robot.find_bodies(eef_names, preserve_order=True)
  for i, eef_name in enumerate(eef_names):
    if i >= len(eef_indices) or eef_indices[i] < 0:
      print(f"Warning: End effector body '{eef_name}' not found in robot body names")
      if i >= len(eef_indices):
        eef_indices.append(-1)
      else:
        eef_indices[i] = -1

  # Store object size for distance method (will be set during first frame if using distance method)
  object_size_global = None

  log: dict[str, Any] = {
    "fps": [int(round(output_fps))],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
  }
  if log_object:
    log["object_pos_w"] = []
    log["object_quat_w"] = []
    log["object_lin_vel_w"] = []
    log["object_ang_vel_w"] = []
    # Contact information
    log["contact_positions"] = []  # (T, num_eefs, 3)
    log["contact_indicators"] = []  # (T, num_eefs)

  frames_total = getattr(motion, "output_frames", motion.input_frames)
  pbar = tqdm(total=frames_total, desc="Converting", unit="frame", ncols=100)

  scene.reset()
  file_saved = False

  # Track previous object position from motion data for velocity computation
  prev_obj_pos_motion = None
  prev_obj_rot_motion = None

  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = quat_apply_inverse(motion_base_rot, motion_base_ang_vel)
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, :] = motion_dof_pos
    joint_vel[:, :] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    if log_object:
      # Ensure object data exists
      if not hasattr(motion, "object_poss") or not hasattr(motion, "object_rots"):
        raise RuntimeError(
          "Object data not found: object_poss or object_rots missing. "
          "This should not happen if has_object is True."
        )
      if reset_flag:
        frame_idx = (frames_total - 1) if frames_total > 0 else 0
      else:
        frame_idx = motion.current_idx - 1
      # Ensure frame_idx is within bounds
      frame_idx = max(0, min(frame_idx, motion.object_poss.shape[0] - 1))

      # Get current object position from motion data (before adding env_origin)
      curr_obj_pos_motion = motion.object_poss[frame_idx : frame_idx + 1].clone()
      curr_obj_rot_motion = motion.object_rots[frame_idx : frame_idx + 1].clone()

      if obj_pos_offset is not None:
        curr_obj_pos_motion = curr_obj_pos_motion - obj_pos_offset.unsqueeze(0)

      if obj_rot_offset is not None:
        curr_obj_rot_motion = _normalize_quat(
          quat_mul(curr_obj_rot_motion, obj_rot_offset.unsqueeze(0))
        )

      # Compute velocity from motion data positions
      if prev_obj_pos_motion is not None:
        # Linear velocity: (current_pos - prev_pos) / dt
        obj_lin_vel_motion = (
          curr_obj_pos_motion[0] - prev_obj_pos_motion
        ) / sim_cfg.mujoco.timestep

        # Angular velocity: compute from quaternion difference
        q_prev = prev_obj_rot_motion
        q_curr = curr_obj_rot_motion[0]
        q_rel = quat_mul(q_curr, quat_conjugate(q_prev))
        obj_ang_vel_motion = axis_angle_from_quat(q_rel) / sim_cfg.mujoco.timestep
      else:
        # First frame: use zero velocity
        obj_lin_vel_motion = torch.zeros(3, device=device, dtype=torch.float32)
        obj_ang_vel_motion = torch.zeros(3, device=device, dtype=torch.float32)

      # Object position should be in world coordinates, add env_origin offset
      obj_pos_slice = curr_obj_pos_motion.clone()
      obj_pos_slice[:, :2] += scene.env_origins[:, :2]

      obj_pose = torch.cat([obj_pos_slice, curr_obj_rot_motion], dim=-1)
      assert (
        box is not None
      )  # Type narrowing: box is guaranteed to be not None when log_object is True
      box.write_root_link_pose_to_sim(obj_pose)

      # Store current position for next iteration
      prev_obj_pos_motion = curr_obj_pos_motion[0].clone()
      prev_obj_rot_motion = curr_obj_rot_motion[0].clone()

      # Initialize velocities for this frame (always defined in both branches above)
      obj_lin_vel_motion_frame = obj_lin_vel_motion
      obj_ang_vel_motion_frame = obj_ang_vel_motion
    else:
      # Initialize dummy values when log_object is False (shouldn't be used)
      obj_lin_vel_motion_frame = torch.zeros(3, device=device, dtype=torch.float32)
      obj_ang_vel_motion_frame = torch.zeros(3, device=device, dtype=torch.float32)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)

    # Extract data for all joints
    log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
    log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())

    # Extract data for ALL bodies (mjlab expects complete body data)
    body_pos_w = robot.data.body_link_pos_w[0, :].cpu().numpy().copy()
    log["body_pos_w"].append(body_pos_w)
    log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
    log["body_lin_vel_w"].append(
      robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
    )
    log["body_ang_vel_w"].append(
      robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
    )

    if log_object:
      # Get object position from simulation (after setting pose and forward step)
      assert (
        box is not None
      )  # Type narrowing: box is guaranteed to be not None when log_object is True
      curr_obj_pos = box.data.body_link_pos_w[0, 0].cpu().numpy().copy()
      curr_obj_rot = box.data.body_link_quat_w[0, 0].cpu().numpy().copy()

      log["object_pos_w"].append(curr_obj_pos)
      log["object_quat_w"].append(curr_obj_rot)

      # Use velocities computed from motion data (computed earlier in the loop)
      obj_lin_vel = obj_lin_vel_motion_frame.cpu().numpy()
      obj_ang_vel = obj_ang_vel_motion_frame.cpu().numpy()

      log["object_lin_vel_w"].append(obj_lin_vel)
      log["object_ang_vel_w"].append(obj_ang_vel)

      # Use distance threshold method (simpler fallback)
      # Get end effector positions from site_pos_w
      # site_pos_w = robot.data.site_pos_w[0, :].cpu().numpy().copy()
      # eef_positions = np.array(
      #   [
      #     site_pos_w[idx] if idx >= 0 else np.array([np.nan, np.nan, np.nan])
      #     for idx in eef_indices
      #   ]
      # )
      body_pos_w = (
        robot.data.body_link_pos_w[0, :, :3].cpu().numpy().copy()
      )  # (num_bodies, 3)
      eef_positions = np.array(
        [
          body_pos_w[idx] if idx >= 0 else np.array([np.nan, np.nan, np.nan])
          for idx in eef_indices
        ]
      )

      # Get object size from MuJoCo model (half-extents for box geometry)
      object_size = None
      if box is not None:
        # Try to find the object body - check common names
        object_body_names = ["largebox_link", "box", "object"]
        object_body_id = -1
        for body_name in object_body_names:
          body_id = mujoco.mj_name2id(sim.mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
          if body_id >= 0:
            object_body_id = int(body_id)
            break

        if object_body_id >= 0:
          # Find geoms belonging to this body
          for geom_id in range(sim.mj_model.ngeom):
            if int(sim.mj_model.geom_bodyid[geom_id]) == object_body_id:
              # Check if it's a box geometry
              if sim.mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                object_size = sim.mj_model.geom_size[geom_id].copy()  # Half-extents
                break

        # If still not found, try to get from box entity's first geom
        if object_size is None and hasattr(box, "spec"):
          # Try to get geom size from the entity's spec
          try:
            geom_ids = box.indexing.geom_ids
            if len(geom_ids) > 0:
              geom_id_val = (
                geom_ids[0].item() if hasattr(geom_ids[0], "item") else int(geom_ids[0])
              )
              geom_id = int(geom_id_val)  # Ensure it's an int for indexing
              if sim.mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
                object_size = sim.mj_model.geom_size[geom_id].copy()
          except Exception:
            pass

      # Store object size globally (only print once)
      if object_size_global is None:
        object_size_global = object_size
        if object_size is not None:
          print(f"[Contact Detection] Object size (half-extents): {object_size}")
          print(
            f"[Contact Detection] Effective bounding box: ±{object_size + contact_threshold} (size + threshold)"
          )
        else:
          print(
            f"[Contact Detection] Warning: Object size not found, using simple distance threshold ({contact_threshold}m) from center"
          )

      contact_data = extract_contacts_from_distance(
        eef_positions,
        curr_obj_pos,
        object_rotation=curr_obj_rot,
        object_size=object_size,
        threshold=contact_threshold,
      )

      log["contact_positions"].append(contact_data["contact_positions"])
      log["contact_indicators"].append(contact_data["contact_indicators"])

    pbar.update(1)
    if reset_flag:
      file_saved = True

  pbar.close()

  # Stack arrays
  for k in (
    "joint_pos",
    "joint_vel",
    "body_pos_w",
    "body_quat_w",
    "body_lin_vel_w",
    "body_ang_vel_w",
  ):
    log[k] = np.stack(log[k], axis=0)

  if log_object and len(log["object_pos_w"]) > 0:
    log["object_pos_w"] = np.stack(log["object_pos_w"], axis=0)
    log["object_quat_w"] = np.stack(log["object_quat_w"], axis=0)
    log["object_lin_vel_w"] = np.stack(log["object_lin_vel_w"], axis=0)
    log["object_ang_vel_w"] = np.stack(log["object_ang_vel_w"], axis=0)
    # Stack contact information
    if len(log["contact_positions"]) > 0:
      log["contact_positions"] = np.stack(
        log["contact_positions"], axis=0
      )  # (T, num_eefs, 3)
      log["contact_indicators"] = np.stack(
        log["contact_indicators"], axis=0
      )  # (T, num_eefs)

      # Print contact extraction summary
      print(f"\n{'=' * 60}")
      print("Contact Extraction Summary")
      print(f"{'=' * 60}")
      print(f"Method: {contact_method}")
      if contact_method == "distance":
        print(f"Distance threshold: {contact_threshold}m")

      print("\nContact data shape:")
      print(f"  Positions: {log['contact_positions'].shape} (T, num_eefs, 3)")
      print(f"  Indicators: {log['contact_indicators'].shape} (T, num_eefs)")

      num_frames = log["contact_indicators"].shape[0]
      num_eefs = log["contact_indicators"].shape[1]

      # Print diagnostic info for distance method if no contacts found
      if contact_method == "distance" and log["contact_indicators"].sum() == 0:
        print("\n⚠️  Diagnostic information (no contacts detected):")
        print(
          "  Note: End effector positions are extracted from palm sites, not body positions."
        )
        print("  Check that palm sites are correctly positioned in the robot model.")

      print("\nPer-end-effector contact statistics:")
      for i, eef_name in enumerate(eef_names):
        if i < num_eefs:
          contact_count = log["contact_indicators"][:, i].sum()
          contact_percentage = (
            100.0 * contact_count / num_frames if num_frames > 0 else 0.0
          )
          print(f"  {eef_name}:")
          print(
            f"    Contact frames: {contact_count}/{num_frames} ({contact_percentage:.1f}%)"
          )

          # Show contact positions for frames with contact
          contact_frames = np.where(log["contact_indicators"][:, i])[0]
          if len(contact_frames) > 0:
            # Show first few and last few contact positions
            num_samples = min(5, len(contact_frames))
            print(f"    Sample contact positions (first {num_samples}):")
            for frame_idx in contact_frames[:num_samples]:
              pos = log["contact_positions"][frame_idx, i]
              if not np.isnan(pos).any():
                print(
                  f"      Frame {frame_idx}: [{pos[0]:.4f}, {pos[1]:.4f}, {pos[2]:.4f}]"
                )

            if len(contact_frames) > num_samples:
              print(
                f"    ... and {len(contact_frames) - num_samples} more contact frames"
              )

      # Overall statistics
      total_contacts = log["contact_indicators"].sum()
      max_possible = num_frames * num_eefs
      overall_percentage = (
        100.0 * total_contacts / max_possible if max_possible > 0 else 0.0
      )
      print("\nOverall statistics:")
      print(
        f"  Total contact detections: {total_contacts}/{max_possible} ({overall_percentage:.1f}%)"
      )
      print(
        f"  Frames with at least one contact: {(log['contact_indicators'].sum(axis=1) > 0).sum()}/{num_frames}"
      )
      print(
        f"  Frames with both contacts: {(log['contact_indicators'].sum(axis=1) == num_eefs).sum()}/{num_frames}"
      )
      print(f"{'=' * 60}\n")

  np.savez(output_file, **log)  # type: ignore[arg-type]
  print(f"\nSaved mjlab motion to: {output_file}")
  print(
    f"Output contains {log['body_pos_w'].shape[0]} frames with {log['body_pos_w'].shape[1]} body links"
  )
  print(f"  All robot bodies: {robot.body_names}")

  print("Uploading to Weights & Biases...")
  import wandb

  COLLECTION = output_name
  run = wandb.init(project="victor_motions", name=COLLECTION, entity="ATARITUM")
  print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
  REGISTRY = "motions"
  logged_artifact = run.log_artifact(
    artifact_or_path=output_file, name=COLLECTION, type=REGISTRY
  )
  run.link_artifact(
    artifact=logged_artifact,
    target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}",
  )
  print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")
  wandb.finish()


def main(
  input_file: str,
  output_name: str,
  repeat_last_frame: int = 0,
  repeat_first_frame: int = 0,
  start_frames: int = 0,
  start_easing: str = "smoothstep",
  append_reverse: bool = False,
  extract_object_states: bool = True,
  speed: float = 1.0,
  contact_threshold: float = 0.05,
  contact_method: str = "distance",
  output_dir: str = "motions/output/",
  device: str = "cuda:0",
  standup_frames: int = 0,
  standup_base_height: float | None = None,
  standup_preserve_xy: bool = True,
  standup_easing: str = "smoothstep",
  object_position_offset: tuple[float, float, float] | None = None,
  object_rotation_offset: tuple[float, float, float, float] | None = None,
  align_first_frame: bool = True,
):
  """Convert trajectory NPZ file to mjlab motion format with contact extraction.

  Args:
    input_file: Path to input NPZ file
    output_dir: Base output directory
    output_name: Name for the output (used to create motions/output/<output_name>/motion.npz)
    speed: Speed multiplier for trajectory (output_fps = input_fps * speed)
    device: Device to use for simulation
    repeat_last_frame: Number of times to repeat the last frame
    repeat_first_frame: Number of times to repeat the first frame
    append_reverse: If True, append the reversed motion trajectory at the end
    extract_object_states: If False, skip extracting object states and contact information
    contact_method: Method to extract contacts - "mujoco" (preferred) or "distance"
    contact_threshold: Distance threshold in meters for "distance" method
    standup_frames: Number of tail frames to blend to nominal stand pose (0 disables)
    standup_base_height: Optional override for final base height (None keeps nominal)
    standup_preserve_xy: Keep last XY position when transitioning to stand pose
    standup_easing: Easing function used when blending into the stand pose
    object_position_offset: Optional constant XYZ offset (meters) applied to tracked object
    object_rotation_offset: Optional constant quaternion (w, x, y, z) applied to tracked object
    align_first_frame: Translate XY and rotate yaw so first frame aligns with world axes

  Example usage:
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run src/mjlab/scripts/victor_npz_to_npz.py
    --output-dir motions/output/
    --speed 1.4
    --repeat-last-frame 150 --repeat-first-frame 100
    --input-file motions/input/time_x_u_traj_rl_format.npz
    --output-name victor_object_repeated
    --append-reverse
  """
  convert(
    input_file=input_file,
    output_dir=output_dir,
    output_name=output_name,
    speed=speed,
    device=device,
    repeat_last_frame=repeat_last_frame,
    repeat_first_frame=repeat_first_frame,
    start_frames=start_frames,
    start_easing=start_easing,
    append_reverse=append_reverse,
    extract_object_states=extract_object_states,
    contact_method=contact_method,
    contact_threshold=contact_threshold,
    standup_frames=standup_frames,
    standup_base_height=standup_base_height,
    standup_preserve_xy=standup_preserve_xy,
    standup_easing=standup_easing,
    object_position_offset=object_position_offset,
    object_rotation_offset=object_rotation_offset,
    align_first_frame=align_first_frame,
  )


if __name__ == "__main__":
  tyro.cli(main)