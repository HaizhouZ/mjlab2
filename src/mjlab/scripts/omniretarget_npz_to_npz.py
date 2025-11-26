import os
import shutil
from pathlib import Path
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


class OmniRetargetTrajectoryLoader:
  def __init__(
    self,
    input_file: str,
    output_fps: int,
    device: torch.device | str,
    repeat_last_frame: int = 0,
    repeat_first_frame: int = 0,
    append_reverse: bool = False,
  ):
    self.input_file = input_file
    self.output_fps = int(output_fps)
    self.output_dt = 1.0 / float(self.output_fps)
    self.device = device
    self.current_idx = 0
    self.repeat_last_frame = max(0, int(repeat_last_frame))
    self.repeat_first_frame = max(0, int(repeat_first_frame))
    self.append_reverse = bool(append_reverse)

    self._load()
    self._resample_to_output_fps_using_times()
    self._repeat_first_frame()
    self._repeat_last_frame()
    # Compute velocities before appending reverse so we can properly reverse and negate them
    self._compute_velocities_from_resampled()
    self._append_reverse()

  def _load(self) -> None:
    """Load OmniRetarget format NPZ file.

    Format: qpos (T, 43)
      - qpos[:, 0:7] = base quat (wxyz) + base pos (xyz)
      - qpos[:, 7:36] = joint positions (29D)
      - qpos[:, 36:43] = object quat (wxyz) + object pos (xyz)
    """
    data = np.load(self.input_file, allow_pickle=True)
    print(f"[Loader] Loading OmniRetarget format from: {self.input_file}")

    assert "qpos" in data, (
      f"Key 'qpos' not found in OmniRetarget NPZ file: {self.input_file}"
    )
    qpos_np = data["qpos"].astype(np.float32)  # (T, 43)

    # Get fps if available, otherwise infer from data
    if "fps" in data:
      input_fps = int(data["fps"])
    else:
      # Default to 30 fps if not specified
      input_fps = 30
      print(f"[Loader] Warning: 'fps' not found, defaulting to {input_fps} fps")

    T = qpos_np.shape[0]

    # Build time array from fps
    times_np = np.arange(T, dtype=np.float32) * (1.0 / input_fps)

    self.input_times_np = times_np
    self.input_times = torch.from_numpy(times_np).to(self.device)
    self.input_frames = T

    # Extract base pose: qpos[:, 0:7] = [qw, qx, qy, qz, x, y, z]
    self.motion_base_rots_input = _normalize_quat(
      torch.from_numpy(qpos_np[:, 0:4]).to(self.device)
    )
    self.motion_base_poss_input = torch.from_numpy(qpos_np[:, 4:7]).to(self.device)

    # Extract joint positions: qpos[:, 7:36] (29D)
    self.motion_dof_poss_input = torch.from_numpy(qpos_np[:, 7:36]).to(self.device)

    # Extract object pose: qpos[:, 36:43] = [qw, qx, qy, qz, x, y, z]
    self.has_object = True
    self.object_rots_input = _normalize_quat(
      torch.from_numpy(qpos_np[:, 36:40]).to(self.device)
    )
    self.object_pos_input = torch.from_numpy(qpos_np[:, 40:43]).to(self.device)

    # Timing setup
    self.duration = float(times_np[-1] - times_np[0]) if T > 1 else 0.0
    self.input_dt = float(self.duration / max(1, T - 1)) if T > 1 else 1.0 / input_fps
    self.input_fps = input_fps

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
  output_fps: float = 50.0,
  device: str = "cuda:0",
  repeat_last_frame: int = 0,
  repeat_first_frame: int = 0,
  append_reverse: bool = False,
  extract_object_states: bool = True,
  contact_method: str = "distance",
  contact_threshold: float = 0.05,
) -> tuple[dict[str, int] | None, str]:
  """Convert OmniRetarget NPZ file to mjlab motion.npz using simulation states.

  Extracts data for all robot bodies to match the expected mjlab motion format.
  The OmniRetarget format has qpos with 43 dimensions:
    - qpos[:, 0:7] = base quat (wxyz) + base pos (xyz)
    - qpos[:, 7:36] = joint positions (29D)
    - qpos[:, 36:43] = object quat (wxyz) + object pos (xyz)

  When object data is present, also extracts contact information for end effectors.

  Args:
    input_file: Path to input NPZ file
    output_dir: Base output directory
    output_name: Name for the output (used to create motions/output/<output_name>/motion.npz)
    output_fps: Output frame rate
    device: Device to use for simulation
    repeat_last_frame: Number of times to repeat the last frame
    repeat_first_frame: Number of times to repeat the first frame
    append_reverse: If True, append the reversed motion trajectory at the end
    extract_object_states: If False, skip extracting object states and contact information
    contact_method: Method to extract contacts - "mujoco" (preferred, uses physics) or "distance" (uses threshold)
    contact_threshold: Distance threshold in meters for "distance" method (default: 0.05m = 5cm)
  """
  # Construct output path: motions/output/<output_name>/motion.npz
  output_path = f"motions/output/{output_name}/motion.npz"
  os.makedirs(os.path.dirname(output_path), exist_ok=True)
  output_file = output_path

  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / float(output_fps)
  if extract_object_states:
    scene = Scene(unitree_g1_flat_tracking_env_cfg_box().scene, device=device)
  else:
    scene = Scene(unitree_g1_flat_tracking_env_cfg().scene, device=device)
  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  motion = OmniRetargetTrajectoryLoader(
    input_file=input_file,
    output_fps=int(round(output_fps)),
    device=sim.device,
    repeat_last_frame=repeat_last_frame,
    repeat_first_frame=repeat_first_frame,
    append_reverse=append_reverse,
  )

  robot: Entity = scene["robot"]
  box: Entity | None = scene.entities.get("box") if hasattr(scene, "entities") else None
  log_object = (
    extract_object_states
    and getattr(motion, "has_object", False)
    and (box is not None and not box.data.is_fixed_base)
  )

  # Extract ALL bodies - mjlab expects complete body data for all robot bodies
  # The body_names in the config are used for tracking/observation, but motion files need all bodies
  print(f"Robot has {len(robot.body_names)} bodies: {robot.body_names}")

  # End effector body names for contact detection
  eef_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

  # Get end effector body indices in robot.body_names
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
      # Get end effector positions from body_link_pos_w
      body_pos_w = (
        robot.data.body_link_pos_w[0, :, :3].cpu().numpy().copy()
      )  # (num_bodies, 3)
      eef_positions = np.array(
        [
          body_pos_w[idx] if idx >= 0 else np.array([np.nan, np.nan, np.nan])
          for idx in eef_indices
        ]
      )

      # Get object size from MuJoCo model (half-extents for box geometry or mesh bounding box)
      object_size = None
      if box is not None:
        # Try to find the object body - check common names
        object_body_names = ["largebox", "largebox_link", "box", "object"]
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
              geom_type = sim.mj_model.geom_type[geom_id]
              # Check if it's a box geometry
              if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                object_size = sim.mj_model.geom_size[geom_id].copy()  # Half-extents
                break
              # Check if it's a mesh geometry - compute bounding box from mesh vertices
              elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                try:
                  # Get mesh ID for this geometry
                  mesh_id = int(sim.mj_model.geom_dataid[geom_id])
                  # Get mesh vertex data
                  vert_start = int(sim.mj_model.mesh_vertadr[mesh_id])
                  vert_count = int(sim.mj_model.mesh_vertnum[mesh_id])
                  # Extract vertices
                  vertices = sim.mj_model.mesh_vert[
                    vert_start : vert_start + vert_count
                  ]  # (vert_count, 3)
                  # Compute bounding box (half-extents)
                  if vertices.shape[0] > 0:
                    min_bounds = vertices.min(axis=0)
                    max_bounds = vertices.max(axis=0)
                    # Half-extents = (max - min) / 2
                    object_size = ((max_bounds - min_bounds) / 2.0).copy()
                    break
                except Exception as e:
                  print(
                    f"[Contact Detection] Warning: Failed to extract mesh bounding box: {e}"
                  )
                  pass

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
              geom_type = sim.mj_model.geom_type[geom_id]
              if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
                object_size = sim.mj_model.geom_size[geom_id].copy()
              elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
                # Compute bounding box from mesh vertices
                try:
                  mesh_id = int(sim.mj_model.geom_dataid[geom_id])
                  vert_start = int(sim.mj_model.mesh_vertadr[mesh_id])
                  vert_count = int(sim.mj_model.mesh_vertnum[mesh_id])
                  vertices = sim.mj_model.mesh_vert[
                    vert_start : vert_start + vert_count
                  ]
                  if vertices.shape[0] > 0:
                    min_bounds = vertices.min(axis=0)
                    max_bounds = vertices.max(axis=0)
                    object_size = ((max_bounds - min_bounds) / 2.0).copy()
                except Exception as e:
                  print(
                    f"[Contact Detection] Warning: Failed to extract mesh bounding box from entity: {e}"
                  )
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
        print("  Note: End effector positions are extracted from body link positions.")
        print("  Check that the contact threshold and object size are appropriate.")

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

  # Return contact statistics for filename modification
  contact_stats = None
  if log_object and "contact_indicators" in log:
    contact_indicators = log["contact_indicators"]
    # Check if it's already a numpy array (stacked) or still a list
    if isinstance(contact_indicators, np.ndarray) and contact_indicators.shape[0] > 0:
      num_frames = contact_indicators.shape[0]
      num_eefs = contact_indicators.shape[1]
      frames_with_both_contacts = (contact_indicators.sum(axis=1) == num_eefs).sum()
      contact_stats = {
        "frames_with_both_contacts": int(frames_with_both_contacts),
        "total_frames": int(num_frames),
      }

  return contact_stats, output_file


def main(
  input_dir: str = "/home/breakout/workspace/OmniRetarget_Dataset/robot-object",
  output_name: str | None = None,
  input_file: str | None = None,
  convert_all: bool = False,
  repeat_last_frame: int = 0,
  repeat_first_frame: int = 0,
  append_reverse: bool = False,
  extract_object_states: bool = True,
  output_fps: float = 50.0,
  contact_threshold: float = 0.08,
  contact_method: str = "distance",
  output_dir: str = "motions/output/",
  device: str = "cuda:0",
):
  """Convert OmniRetarget NPZ files to mjlab motion format with contact extraction.

  Args:
    input_dir: Directory containing OmniRetarget NPZ files
    output_name: Name for the output directory
                - Single file: creates motions/output/<output_name>/motion.npz
                - Convert all with output_name: creates motions/output/<output_name>/<filename>/motion.npz
                - Convert all without output_name: creates motions/output/<filename>/motion.npz
                - If None and input_file is provided, will derive from input_file name
    input_file: Specific file to convert (if None, will convert first file in directory for testing)
    convert_all: If True, convert all files in the directory. If False, only convert one file for testing.
    output_dir: Base output directory
    output_fps: Output frame rate
    device: Device to use for simulation
    repeat_last_frame: Number of times to repeat the last frame
    repeat_first_frame: Number of times to repeat the first frame
    append_reverse: If True, append the reversed motion trajectory at the end
    extract_object_states: If False, skip extracting object states and contact information
    contact_method: Method to extract contacts - "mujoco" (preferred) or "distance"
    contact_threshold: Distance threshold in meters for "distance" method

  Example usage:
    # Convert one file for testing
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run src/mjlab/scripts/omniretarget_npz_to_npz.py
    --input-file /home/breakout/workspace/OmniRetarget_Dataset/robot-object/sub10_largebox_000_original.npz
    --output-name test_omniretarget

    # Convert all files (each in its own directory)
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run src/mjlab/scripts/omniretarget_npz_to_npz.py
    --input-dir /home/breakout/workspace/OmniRetarget_Dataset/robot-object
    --convert-all

    # Convert all files into a common parent directory
    MUJOCO_GL=egl CUDA_VISIBLE_DEVICES=0 uv run src/mjlab/scripts/omniretarget_npz_to_npz.py
    --input-dir /home/breakout/workspace/OmniRetarget_Dataset/robot-object
    --convert-all --output-name omniretarget
    # Creates: motions/output/omniretarget/sub10_largebox_000_original/motion.npz, etc.
  """
  # Determine which files to convert
  if input_file is not None:
    # Convert specific file
    files_to_convert = [input_file]
    if output_name is None:
      # Derive output name from input file name
      output_name = Path(input_file).stem
  else:
    # Get all NPZ files from directory
    input_path = Path(input_dir)
    all_files = sorted(input_path.glob("*.npz"))

    if len(all_files) == 0:
      print(f"Error: No NPZ files found in {input_dir}")
      return

    if convert_all:
      files_to_convert = [str(f) for f in all_files]
      print(f"Found {len(files_to_convert)} files to convert")
    else:
      # Just convert the first file for testing
      files_to_convert = [str(all_files[0])]
      print(f"Testing with first file: {files_to_convert[0]}")
      print("To convert all files, use --convert-all flag")
      if output_name is None:
        output_name = all_files[0].stem

  # Convert each file
  for file_path in files_to_convert:
    # Determine output name first
    if len(files_to_convert) > 1:
      # Converting multiple files
      if output_name is not None:
        # Use output_name as parent directory: output_name/filename
        file_output_name = f"{output_name}/{Path(file_path).stem}"
      else:
        # No parent directory specified, use filename directly
        file_output_name = Path(file_path).stem
    else:
      # Converting single file
      if output_name is None:
        # Derive output name from file name
        file_output_name = Path(file_path).stem
      else:
        # Use provided output name
        file_output_name = output_name

    # CHECK FIRST: Skip if already converted (before doing any processing)
    initial_output_path = f"motions/output/{file_output_name}/motion.npz"

    # Check if initial output exists
    if os.path.exists(initial_output_path):
      print(f"\n{'=' * 60}")
      print(f"SKIPPING (already converted): {file_path}")
      print(f"Output exists: {initial_output_path}")
      print(f"{'=' * 60}\n")
      continue

    # Also check if a renamed version exists (with contact stats format)
    # Pattern: <digits>-<digits>-<filename>
    # Split the output path to get parent directory and filename
    output_parts = file_output_name.split("/")
    if len(output_parts) > 1:
      # Nested path like "omniretarget/sub10_largebox_000_original"
      parent_dir_path = os.path.join("motions", "output", "/".join(output_parts[:-1]))
      filename_base = output_parts[-1]
    else:
      # Flat path like "sub10_largebox_000_original"
      parent_dir_path = "motions/output"
      filename_base = output_parts[0]

    # Check parent directory for renamed versions
    already_converted = False
    if os.path.exists(parent_dir_path) and os.path.isdir(parent_dir_path):
      for item in os.listdir(parent_dir_path):
        item_path = os.path.join(parent_dir_path, item)
        if os.path.isdir(item_path):
          # Check if it matches the pattern: <digits>-<digits>-<filename_base>
          parts = item.split("-")
          if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
            # Extract the filename part
            potential_filename = "-".join(parts[2:])
            if potential_filename == filename_base:
              # Check if motion.npz exists in this directory
              potential_output = os.path.join(item_path, "motion.npz")
              if os.path.exists(potential_output):
                print(f"\n{'=' * 60}")
                print(f"SKIPPING (already converted): {file_path}")
                print(f"Output exists: {potential_output}")
                print(f"{'=' * 60}\n")
                already_converted = True
                break

    if already_converted:
      continue

    print(f"\n{'=' * 60}")
    print(f"Converting: {file_path}")
    print(f"Output name: {file_output_name}")
    print(f"{'=' * 60}\n")

    contact_stats, output_file = convert(
      input_file=file_path,
      output_dir=output_dir,
      output_name=file_output_name,
      output_fps=output_fps,
      device=device,
      repeat_last_frame=repeat_last_frame,
      repeat_first_frame=repeat_first_frame,
      append_reverse=append_reverse,
      extract_object_states=extract_object_states,
      contact_method=contact_method,
      contact_threshold=contact_threshold,
    )

    # Rename output directory to include contact statistics if available
    if contact_stats is not None:
      frames_with_both = contact_stats["frames_with_both_contacts"]
      total_frames = contact_stats["total_frames"]

      # Extract the current output directory path
      current_output_dir = os.path.dirname(output_file)  # motions/output/<path>
      # Split into base and leaf directory
      path_parts = current_output_dir.split(os.sep)
      base_output_dir = os.sep.join(
        path_parts[:-1]
      )  # motions/output or motions/output/parent
      current_name = path_parts[-1]  # leaf directory name

      # Check if current_name already has the format (to avoid double-appending)
      # Format: <frames>-<total>-<filename>
      parts = current_name.split("-")
      if len(parts) >= 3 and parts[0].isdigit() and parts[1].isdigit():
        # Already has the format, extract just the filename part
        original_name = "-".join(parts[2:])
      else:
        # Doesn't have the format yet, use current_name as base
        original_name = current_name

      # Construct new name: <frames_with_both>-<total_frames>-<original_name>
      new_name = f"{frames_with_both}-{total_frames}-{original_name}"
      new_output_dir = os.path.join(base_output_dir, new_name)
      new_output_file = os.path.join(new_output_dir, "motion.npz")

      # Only rename if the name actually changed
      if current_output_dir != new_output_dir:
        os.makedirs(new_output_dir, exist_ok=True)
        # Move the file
        shutil.move(output_file, new_output_file)
        # Remove old directory if empty
        try:
          os.rmdir(current_output_dir)
        except OSError:
          pass  # Directory not empty or doesn't exist, that's fine

        print("\nRenamed output directory to include contact statistics:")
        print(f"  Old: {current_output_dir}")
        print(f"  New: {new_output_dir}")
        print(
          f"  Format: <{frames_with_both} frames with both contacts>-<{total_frames} total frames>-<{original_name}>"
        )


if __name__ == "__main__":
  tyro.cli(main)
