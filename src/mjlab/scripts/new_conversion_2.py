import os
from typing import Any

import mujoco
import numpy as np
import torch
import tyro
from tqdm import tqdm

from mjlab.entity import Entity
from mjlab.scene import Scene
from mjlab.sensor.contact_sensor import ContactSensor, ContactSensorCfg
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.registry import load_env_cfg
from mjlab.utils.lab_api.math import (
  axis_angle_from_quat,
  quat_apply,
  quat_apply_inverse,
  quat_conjugate,
  quat_mul,
  quat_slerp,
)
from mjlab.viewer.offscreen_renderer import OffscreenRenderer
from mjlab.viewer.viewer_config import ViewerConfig

CONTACT_THRESHOLD = 0.06  # Used for distance method
EE_SITE_NAMES = ["left_palm", "right_palm", "left_foot_tip", "right_foot_tip"]
CONTACT_EXTRACTION_METHOD = "mujoco"  # "mujoco" or "distance"


def get_object_size(
  mj_model: mujoco.MjModel,
  object_entity: Entity | None,
) -> np.ndarray | None:
  """Get object size (half-extents) from MuJoCo model for box geometry.

  Args:
    mj_model: MuJoCo model containing the object.
    object_entity: Entity object representing the tracked object, or None.

  Returns:
    Object size as (3,) array of half-extents, or None if not found.
  """
  if object_entity is None:
    return None

  object_size = None
  # Try to find the object body - check common names
  object_body_names = ["largebox_link", "box", "object"]
  object_body_id = -1
  for body_name in object_body_names:
    body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id >= 0:
      object_body_id = int(body_id)
      break

  if object_body_id >= 0:
    # Find geoms belonging to this body
    for geom_id in range(mj_model.ngeom):
      if int(mj_model.geom_bodyid[geom_id]) == object_body_id:
        # Check if it's a box geometry
        if mj_model.geom_type[geom_id] in [
          mujoco.mjtGeom.mjGEOM_BOX,
          mujoco.mjtGeom.mjGEOM_CYLINDER,
        ]:
          object_size = mj_model.geom_size[geom_id].copy()  # Half-extents
          break

  # If still not found, try to get from object entity's first geom
  if object_size is None and hasattr(object_entity, "indexing"):
    # Try to get geom size from the entity's spec
    try:
      geom_ids = object_entity.indexing.geom_ids
      if len(geom_ids) > 0:
        geom_id_val = (
          geom_ids[0].item() if hasattr(geom_ids[0], "item") else int(geom_ids[0])
        )
        geom_id = int(geom_id_val)  # Ensure it's an int for indexing
        if mj_model.geom_type[geom_id] in [
          mujoco.mjtGeom.mjGEOM_BOX,
          mujoco.mjtGeom.mjGEOM_CYLINDER,
        ]:
          object_size = mj_model.geom_size[geom_id].copy()
    except Exception:
      pass

  return object_size


def extract_contacts_from_scene_sensors(
  scene: Scene,
  sensor_names: list[str],
) -> dict[str, Any]:
  """Extract contact information from scene ContactSensor instances.

  This uses Warp-native contact sensors, avoiding the need to sync to MuJoCo.

  Args:
    scene: Scene instance with initialized sensors.
    sensor_names: List of sensor names to extract (e.g., ["left_eef_contact", "right_eef_contact"]).

  Returns:
    Dictionary with contact information:
      - contact_positions: (num_sensors, 3) array of contact positions (NaN if no contact)
      - contact_indicators: (num_sensors,) boolean array indicating if contact exists
      - sensor_names: List of sensor names in order
  """
  num_sensors = len(sensor_names)
  contact_indicators = np.zeros(num_sensors, dtype=bool)
  contact_positions = np.full((num_sensors, 3), np.nan, dtype=np.float32)

  for sensor_idx, sensor_name in enumerate(sensor_names):
    if sensor_name in scene.sensors:
      sensor = scene.sensors[sensor_name]
      if isinstance(sensor, ContactSensor):
        contact_data = sensor.data
        # Check if contact found (found > 0 means contact)
        if contact_data.found is not None:
          # found is [B, N] where N is number of primary geoms
          # For single contact sensors, typically N=1
          found = contact_data.found[0]  # [N] for first environment
          has_contact = (found > 0).any().item()
          contact_indicators[sensor_idx] = has_contact

          # Get contact position if available
          if contact_data.pos is not None and has_contact:
            # pos is [B, N, 3], get first contact position from first primary geom
            # Use the contact with the smallest distance (most penetration)
            if contact_data.dist is not None:
              dists = contact_data.dist[0]  # [N]
              valid_mask = dists < 0  # Negative distance means penetration
              if valid_mask.any():
                best_idx = dists[valid_mask].argmin()
                valid_indices = torch.where(valid_mask)[0]
                best_contact_idx = valid_indices[best_idx]
                contact_positions[sensor_idx] = (
                  contact_data.pos[0, best_contact_idx].cpu().numpy()
                )
            else:
              # No distance info, just take first contact
              contact_positions[sensor_idx] = contact_data.pos[0, 0].cpu().numpy()
  # print(f"Contact positions: {contact_positions}")
  # print(f"Contact indicators: {contact_indicators}")
  # print("================================================")
  return {
    "contact_positions": contact_positions,
    "contact_indicators": contact_indicators,
    "sensor_names": sensor_names,
  }


def extract_contacts_from_distance(
  eef_positions: np.ndarray,
  object_position: np.ndarray,
  object_rotation: np.ndarray | None = None,
  object_size: np.ndarray | None = None,
  threshold: float = 0.05,
  shape: str = "box",
) -> dict[str, Any]:
  """Extract contact information using distance threshold with object size consideration.

  This is a simpler fallback method that checks if end effector positions are within
  the object's bounds (accounting for size) plus a threshold margin.

  Args:
    eef_positions: (num_eefs, 3) array of end effector positions in world frame
    object_position: (3,) array of object position (center) in world frame
    object_rotation: (4,) array of object quaternion (w, x, y, z) in world frame, or None for axis-aligned
    object_size: (3,) array of object size parameters, or None to use threshold only
                 For "box": half-extents [half_x, half_y, half_z] (MuJoCo convention)
                 For "cylinder": [radius, radius, half_height] where radius is for x/y, half_height is for z-axis
    threshold: Distance threshold in meters for contact detection (default: 0.05m = 5cm)
               If object_size is None, this is used as distance from center.
               If object_size is provided, this is added as margin around the object bounds.
    shape: Shape type, either "box" or "cylinder" (default: "box")

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
    return {
      "contact_positions": contact_positions,
      "contact_indicators": contact_indicators,
    }

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

  if shape == "box":
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

  elif shape == "cylinder":
    # For cylinder: object_size = [radius, radius, half_height]
    # Check if within cylinder: sqrt(x^2 + y^2) <= radius + threshold AND |z| <= half_height + threshold
    radius = object_size[0]  # Radius for x/y plane
    half_height = object_size[1]  # Half-height along z-axis

    for i in range(num_eefs):
      # Skip if position is NaN (end effector not found)
      if np.isnan(eef_pos_local[i]).any():
        continue

      # Check radial distance (x-y plane)
      radial_dist = np.sqrt(eef_pos_local[i][0] ** 2 + eef_pos_local[i][1] ** 2)
      # Check height (z-axis)
      height_dist = np.abs(eef_pos_local[i][2])

      # Check if within cylinder bounds (with threshold margin)
      within_radius = radial_dist <= (radius + threshold)
      within_height = height_dist <= (half_height + threshold)

      if within_radius and within_height:
        contact_indicators[i] = True
        # Find closest point on cylinder surface
        closest_local = eef_pos_local[i].copy()

        # Clamp height to cylinder bounds
        closest_local[2] = np.clip(closest_local[2], -half_height, half_height)

        # Project radial position to cylinder surface
        if radial_dist > 1e-6:  # Avoid division by zero
          # Normalize radial direction and scale to radius
          radial_dir = eef_pos_local[i][:2] / radial_dist
          closest_local[0] = radial_dir[0] * radius
          closest_local[1] = radial_dir[1] * radius
        else:
          # Point is on z-axis, choose arbitrary direction
          closest_local[0] = radius
          closest_local[1] = 0.0

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

  else:
    raise ValueError(f"Unsupported shape: {shape}. Must be 'box' or 'cylinder'.")

  return {
    "contact_positions": contact_positions,
    "contact_indicators": contact_indicators,
  }


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
    motion_idx: int | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self.motion_idx = (
      motion_idx  # For batch processing: which motion to load (None = all or first)
    )
    self.is_batch = False
    self.num_motions = 0
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from CSV or NPZ file."""
    # Detect file format
    is_npz = self.motion_file.endswith(".npz")

    if is_npz:
      # Load NPZ file
      data = np.load(self.motion_file)

      # Check if it's a batch (3D) or single motion (2D)
      if "motion_data" in data:
        motion_data = data["motion_data"]
        if motion_data.ndim == 3:
          # Batch format: (num_motions, num_frames, features)
          self.is_batch = True
          self.num_motions = motion_data.shape[0]
          print(
            f"[Loader] Detected batch format: {self.num_motions} motions, {motion_data.shape[1]} frames each"
          )

          # Use specified motion_idx or default to 0 (first motion)
          if self.motion_idx is None:
            self.motion_idx = 0
          if self.motion_idx >= self.num_motions:
            raise ValueError(
              f"motion_idx {self.motion_idx} >= num_motions {self.num_motions}"
            )

          print(
            f"[Loader] Loading motion {self.motion_idx + 1}/{self.num_motions} for processing/rendering"
          )
          motion = motion_data[self.motion_idx]  # (num_frames, features)
        else:
          # Single motion format: (num_frames, features)
          self.is_batch = False
          self.num_motions = 1
          motion = motion_data
      else:
        raise ValueError(
          f"NPZ file must contain 'motion_data' key. Found keys: {list(data.keys())}"
        )

      motion = torch.from_numpy(motion.astype(np.float32))
    else:
      # Load CSV file
      self.is_batch = False
      self.num_motions = 1
      if self.line_range is None:
        motion = torch.from_numpy(np.loadtxt(self.motion_file, delimiter=","))
      else:
        motion = torch.from_numpy(
          np.loadtxt(
            self.motion_file,
            delimiter=",",
            skiprows=self.line_range[0] - 1,
            max_rows=self.line_range[1] - self.line_range[0] + 1,
          )
        )

    motion = motion.to(torch.float32).to(self.device)
    # motion[:, 2] -= 0.05
    self.motion_base_poss_input = motion[:, :3]
    self.motion_base_rots_input = motion[:, 3:7]
    self.motion_base_rots_input = self.motion_base_rots_input[
      :, [3, 0, 1, 2]
    ]  # convert to wxyz

    # CSV format from new_conversion_1.py:
    # [x, y, z, qx, qy, qz, qw, joint1, ..., jointN, pd_target1, ..., pd_targetN, obj_x, obj_y, obj_z, obj_qx, obj_qy, obj_qz, obj_qw]
    # Format: base(7) + joints(N) + joint_pd_targets(N) + object(7)
    num_cols = motion.shape[1]

    # Assume PD targets are always available
    # Format: base(7) + joints(N) + joint_pd_targets(N) + object(7)
    # Calculate num_joints from: (num_cols - 14) / 2
    remaining_cols = num_cols - 14  # After removing base(7) and object(7)
    num_joints = remaining_cols // 2

    self.has_pd_targets = True

    # Extract components
    joint_start = 7
    joint_end = joint_start + num_joints
    pd_target_start = joint_end
    pd_target_end = pd_target_start + num_joints
    obj_start = pd_target_end

    self.motion_dof_poss_input = motion[:, joint_start:joint_end]
    self.motion_pd_targets_input = motion[:, pd_target_start:pd_target_end]
    self.motion_object_poss_input = motion[:, obj_start : obj_start + 3]
    self.motion_object_rots_input = motion[:, obj_start + 3 : obj_start + 7]
    self.motion_object_rots_input = self.motion_object_rots_input[
      :, [3, 0, 1, 2]
    ]  # convert to wxyz

    self.has_object = True
    file_format = (
      "NPZ (batch)" if (is_npz and self.is_batch) else ("NPZ" if is_npz else "CSV")
    )
    print(
      f"[Loader] {file_format} - columns: {num_cols}, Joints: {num_joints}, PD targets: yes, Object: yes"
    )

    self.input_frames = motion.shape[0]
    self.duration = (self.input_frames - 1) * self.input_dt

  def _interpolate_motion(self):
    """Interpolates the motion to the output fps."""
    times = torch.arange(
      0, self.duration, self.output_dt, device=self.device, dtype=torch.float32
    )
    self.output_frames = times.shape[0]
    index_0, index_1, blend = self._compute_frame_blend(times)
    self.motion_base_poss = self._lerp(
      self.motion_base_poss_input[index_0],
      self.motion_base_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_base_rots = self._slerp(
      self.motion_base_rots_input[index_0],
      self.motion_base_rots_input[index_1],
      blend,
    )
    self.motion_dof_poss = self._lerp(
      self.motion_dof_poss_input[index_0],
      self.motion_dof_poss_input[index_1],
      blend.unsqueeze(1),
    )
    self.motion_pd_targets = self._lerp(
      self.motion_pd_targets_input[index_0],
      self.motion_pd_targets_input[index_1],
      blend.unsqueeze(1),
    )
    if self.has_object:
      self.motion_object_poss = self._lerp(
        self.motion_object_poss_input[index_0],
        self.motion_object_poss_input[index_1],
        blend.unsqueeze(1),
      )
      self.motion_object_rots = self._slerp(
        self.motion_object_rots_input[index_0],
        self.motion_object_rots_input[index_1],
        blend,
      )
    print(
      f"Motion interpolated, input frames: {self.input_frames}, "
      f"input fps: {self.input_fps}, "
      f"output frames: {self.output_frames}, "
      f"output fps: {self.output_fps}"
    )

  def _lerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Linear interpolation between two tensors."""
    return a * (1 - blend) + b * blend

  def _slerp(
    self, a: torch.Tensor, b: torch.Tensor, blend: torch.Tensor
  ) -> torch.Tensor:
    """Spherical linear interpolation between two quaternions."""
    slerped_quats = torch.zeros_like(a)
    for i in range(a.shape[0]):
      slerped_quats[i] = quat_slerp(a[i], b[i], float(blend[i]))
    return slerped_quats

  def _compute_frame_blend(
    self, times: torch.Tensor
  ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Computes the frame blend for the motion."""
    phase = times / self.duration
    index_0 = (phase * (self.input_frames - 1)).floor().long()
    index_1 = torch.minimum(index_0 + 1, torch.tensor(self.input_frames - 1))
    blend = phase * (self.input_frames - 1) - index_0
    return index_0, index_1, blend

  def _compute_velocities(self):
    """Computes the velocities of the motion."""
    self.motion_base_lin_vels = torch.gradient(
      self.motion_base_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_dof_vels = torch.gradient(
      self.motion_dof_poss, spacing=self.output_dt, dim=0
    )[0]
    self.motion_base_ang_vels = self._so3_derivative(
      self.motion_base_rots, self.output_dt
    )
    if self.has_object:
      self.motion_object_lin_vels = torch.gradient(
        self.motion_object_poss, spacing=self.output_dt, dim=0
      )[0]
      self.motion_object_ang_vels = self._so3_derivative(
        self.motion_object_rots, self.output_dt
      )

  def _so3_derivative(self, rotations: torch.Tensor, dt: float) -> torch.Tensor:
    """Computes the derivative of a sequence of SO3 rotations.

    Args:
      rotations: shape (B, 4).
      dt: time step.
    Returns:
      shape (B, 3).
    """
    q_prev, q_next = rotations[:-2], rotations[2:]
    q_rel = quat_mul(q_next, quat_conjugate(q_prev))  # shape (B−2, 4)

    omega = axis_angle_from_quat(q_rel) / (2.0 * dt)  # shape (B−2, 3)
    omega = torch.cat(
      [omega[:1], omega, omega[-1:]], dim=0
    )  # repeat first and last sample
    return omega

  def get_next_state(
    self,
  ) -> tuple[
    tuple[
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor,
      torch.Tensor | None,
      torch.Tensor | None,
      torch.Tensor | None,
      torch.Tensor | None,
    ],
    bool,
  ]:
    """Gets the next state of the motion.

    Returns:
      Tuple containing:
        - Base pos, base rot, base lin vel, base ang vel, dof pos, dof vel
        - PD targets
        - Object pos, object rot, object lin vel, object ang vel (or None if no object)
        - Reset flag
    """
    base_state = (
      self.motion_base_poss[self.current_idx : self.current_idx + 1],
      self.motion_base_rots[self.current_idx : self.current_idx + 1],
      self.motion_base_lin_vels[self.current_idx : self.current_idx + 1],
      self.motion_base_ang_vels[self.current_idx : self.current_idx + 1],
      self.motion_dof_poss[self.current_idx : self.current_idx + 1],
      self.motion_dof_vels[self.current_idx : self.current_idx + 1],
    )
    joint_pd_targets = self.motion_pd_targets[self.current_idx : self.current_idx + 1]

    if self.has_object:
      object_state = (
        self.motion_object_poss[self.current_idx : self.current_idx + 1],
        self.motion_object_rots[self.current_idx : self.current_idx + 1],
        self.motion_object_lin_vels[self.current_idx : self.current_idx + 1],
        self.motion_object_ang_vels[self.current_idx : self.current_idx + 1],
      )
      state = base_state + (joint_pd_targets,) + object_state
    else:
      state = base_state + (joint_pd_targets,) + (None, None, None, None)

    self.current_idx += 1
    reset_flag = False
    if self.current_idx >= self.output_frames:
      self.current_idx = 0
      reset_flag = True
    return state, reset_flag


def process_single_motion_sim(
  sim: Simulation,
  scene: Scene,
  motion: MotionLoader,
  robot: Entity,
  robot_joint_indexes: torch.Tensor | list[int],
  object_entity: Entity | None,
  has_object_in_scene: bool,
  eef_indexes: list[int],
  render: bool,
  renderer: OffscreenRenderer | None,
  contact_threshold: float,
  output_fps: float,
  task_name: str | None = None,
  contact_sensors: dict[str, ContactSensorCfg] | None = None,
) -> tuple[dict[str, Any], list]:
  """Process a single motion through simulation.

  Returns:
    Tuple of (log dictionary, frames list)
  """
  log: dict[str, Any] = {
    "fps": [output_fps],
    "joint_pos": [],
    "joint_vel": [],
    "body_pos_w": [],
    "body_quat_w": [],
    "body_lin_vel_w": [],
    "body_ang_vel_w": [],
    "joint_pd_targets": [],
  }
  if motion.has_object:
    log["object_pos_w"] = []
    log["object_quat_w"] = []
    log["object_lin_vel_w"] = []
    log["object_ang_vel_w"] = []
    log["contact_positions"] = []
    log["contact_indicators"] = []

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
  file_saved = False
  while not file_saved:
    (
      (
        motion_base_pos,
        motion_base_rot,
        motion_base_lin_vel,
        motion_base_ang_vel,
        motion_dof_pos,
        motion_dof_vel,
        motion_pd_targets,
        motion_object_pos,
        motion_object_rot,
        motion_object_lin_vel,
        motion_object_ang_vel,
      ),
      reset_flag,
    ) = motion.get_next_state()

    root_states = robot.data.default_root_state.clone()
    root_states[:, 0:3] = motion_base_pos
    root_states[:, :2] += scene.env_origins[:, :2]
    root_states[:, 3:7] = motion_base_rot
    root_states[:, 7:10] = motion_base_lin_vel
    root_states[:, 10:] = motion_base_ang_vel
    robot.write_root_state_to_sim(root_states)

    joint_pos = robot.data.default_joint_pos.clone()
    joint_vel = robot.data.default_joint_vel.clone()
    joint_pos[:, robot_joint_indexes] = motion_dof_pos
    joint_vel[:, robot_joint_indexes] = motion_dof_vel
    robot.write_joint_state_to_sim(joint_pos, joint_vel)

    if (
      has_object_in_scene
      and object_entity is not None
      and motion_object_pos is not None
      and motion_object_rot is not None
      and motion_object_lin_vel is not None
      and motion_object_ang_vel is not None
    ):
      object_root_state = torch.cat(
        [
          motion_object_pos,
          motion_object_rot,
          motion_object_lin_vel,
          motion_object_ang_vel,
        ],
        dim=-1,
      )
      object_entity.write_root_state_to_sim(object_root_state)

    sim.forward()
    scene.update(sim.mj_model.opt.timestep)
    if render and renderer is not None:
      renderer.update(sim.data)
      frames.append(renderer.render())

    if not file_saved:
      log["joint_pos"].append(robot.data.joint_pos[0, :].cpu().numpy().copy())
      log["joint_vel"].append(robot.data.joint_vel[0, :].cpu().numpy().copy())
      log["body_pos_w"].append(robot.data.body_link_pos_w[0, :].cpu().numpy().copy())
      log["body_quat_w"].append(robot.data.body_link_quat_w[0, :].cpu().numpy().copy())
      log["body_lin_vel_w"].append(
        robot.data.body_link_lin_vel_w[0, :].cpu().numpy().copy()
      )
      log["body_ang_vel_w"].append(
        robot.data.body_link_ang_vel_w[0, :].cpu().numpy().copy()
      )
      log["joint_pd_targets"].append(motion_pd_targets[0].cpu().numpy().copy())

      if (
        motion.has_object
        and motion_object_pos is not None
        and motion_object_rot is not None
        and motion_object_lin_vel is not None
        and motion_object_ang_vel is not None
      ):
        if has_object_in_scene and object_entity is not None:
          curr_obj_pos = object_entity.data.body_link_pos_w[0, 0].cpu().numpy().copy()
          curr_obj_rot = object_entity.data.body_link_quat_w[0, 0].cpu().numpy().copy()

          log["object_pos_w"].append(curr_obj_pos)
          log["object_quat_w"].append(curr_obj_rot)
          log["object_lin_vel_w"].append(
            object_entity.data.body_link_lin_vel_w[0, 0].cpu().numpy().copy()
          )
          log["object_ang_vel_w"].append(
            object_entity.data.body_link_ang_vel_w[0, 0].cpu().numpy().copy()
          )

          if CONTACT_EXTRACTION_METHOD == "mujoco":
            sensor_names = list(contact_sensors.keys()) if contact_sensors else []
            contact_data = extract_contacts_from_scene_sensors(scene, sensor_names)
          else:
            site_pos_w = robot.data.site_pos_w[0, :].cpu().numpy().copy()
            eef_positions = np.array(
              [
                site_pos_w[idx] if idx >= 0 else np.array([np.nan, np.nan, np.nan])
                for idx in eef_indexes
              ]
            )
            object_size = get_object_size(sim.mj_model, object_entity)

            # Determine shape based on task name
            if task_name is not None and "cylinder" in task_name.lower():
              object_geom = "cylinder"
            elif task_name is not None and "box" in task_name.lower():
              object_geom = "box"
            else:
              # Default to box if task_name is None or doesn't contain shape info
              object_geom = "box"
            contact_data = extract_contacts_from_distance(
              eef_positions,
              curr_obj_pos,
              object_rotation=curr_obj_rot,
              object_size=object_size,
              threshold=contact_threshold,
              shape=object_geom,
            )

          log["contact_positions"].append(contact_data["contact_positions"])
          log["contact_indicators"].append(contact_data["contact_indicators"])
        else:
          log["object_pos_w"].append(motion_object_pos[0].cpu().numpy().copy())
          log["object_quat_w"].append(motion_object_rot[0].cpu().numpy().copy())
          log["object_lin_vel_w"].append(motion_object_lin_vel[0].cpu().numpy().copy())
          log["object_ang_vel_w"].append(motion_object_ang_vel[0].cpu().numpy().copy())

      torch.testing.assert_close(
        robot.data.body_link_lin_vel_w[0, 0], motion_base_lin_vel[0]
      )
      torch.testing.assert_close(
        robot.data.body_link_ang_vel_w[0, 0], motion_base_ang_vel[0]
      )

      frame_count += 1
      pbar.update(1)

      if frame_count % 100 == 0:
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        for k in (
          "joint_pos",
          "joint_vel",
          "body_pos_w",
          "body_quat_w",
          "body_lin_vel_w",
          "body_ang_vel_w",
          "joint_pd_targets",
        ):
          log[k] = np.stack(log[k], axis=0)
        if motion.has_object:
          for k in (
            "object_pos_w",
            "object_quat_w",
            "object_lin_vel_w",
            "object_ang_vel_w",
          ):
            log[k] = np.stack(log[k], axis=0)
          if len(log["contact_positions"]) > 0:  # type: ignore[arg-type]
            log["contact_positions"] = np.stack(  # type: ignore[call-overload]
              log["contact_positions"],  # type: ignore[arg-type]
              axis=0,
            )
            log["contact_indicators"] = np.stack(  # type: ignore[call-overload]
              log["contact_indicators"],  # type: ignore[arg-type]
              axis=0,
            )

  return log, frames


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  project_name,
  collection_name,
  render,
  line_range,
  renderer: OffscreenRenderer | None = None,
  contact_threshold: float = 0.05,
  task_name: str | None = None,
  contact_sensors: dict[str, ContactSensorCfg] | None = None,
):
  # First, check if input is batch format
  is_npz = input_file.endswith(".npz")
  is_batch = False
  num_motions = 1

  if is_npz:
    data = np.load(input_file)
    if "motion_data" in data:
      motion_data = data["motion_data"]
      if motion_data.ndim == 3:
        is_batch = True
        num_motions = motion_data.shape[0]
        print(f"[Batch] Detected batch format: {num_motions} motions")

  # Load first motion to get metadata
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
    motion_idx=0 if is_batch else None,  # Load first motion for rendering
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  # Try to get object entity if it exists
  try:
    object_entity: Entity | None = scene.entities.get("object")
  except (KeyError, AttributeError):
    object_entity = None

  has_object_in_scene = object_entity is not None and motion.has_object
  if motion.has_object and not has_object_in_scene:
    print(
      "[Warning] Object data found in CSV but no object entity in scene. Object states will be logged but not simulated."
    )
  elif has_object_in_scene:
    print(
      "[Info] Object entity found in scene. Object states will be simulated and logged."
    )

  eef_indexes, eef_names_found = robot.find_sites(EE_SITE_NAMES, preserve_order=True)
  for i, eef_name in enumerate(EE_SITE_NAMES):
    if i >= len(eef_indexes) or eef_indexes[i] < 0:
      print(f"Warning: End effector site '{eef_name}' not found in robot site names")
      if i >= len(eef_indexes):
        eef_indexes.append(-1)
      else:
        eef_indexes[i] = -1

  if is_batch:
    # Process all motions in batch
    print(f"\nProcessing {num_motions} motions (rendering only first motion)...")
    all_logs = []
    frames = []

    for motion_idx in range(num_motions):
      print(f"\n--- Processing motion {motion_idx + 1}/{num_motions} ---")

      # Load motion
      motion = MotionLoader(
        motion_file=input_file,
        input_fps=input_fps,
        output_fps=output_fps,
        device=sim.device,
        line_range=line_range,
        motion_idx=motion_idx,
      )

      # Only render the first motion
      should_render = render and motion_idx == 0

      # Process motion
      log, motion_frames = process_single_motion_sim(
        sim=sim,
        scene=scene,
        motion=motion,
        robot=robot,
        robot_joint_indexes=robot_joint_indexes,
        object_entity=object_entity,
        has_object_in_scene=has_object_in_scene,
        eef_indexes=eef_indexes,
        render=should_render,
        renderer=renderer,
        contact_threshold=contact_threshold,
        output_fps=output_fps,
        task_name=task_name,
        contact_sensors=contact_sensors,
      )

      all_logs.append(log)
      if should_render:
        frames = motion_frames

    # Stack all logs to preserve batch shape
    print("\nStacking batch data...")
    batch_log: dict[str, Any] = {}
    for key in all_logs[0].keys():
      if key == "fps":
        batch_log[key] = all_logs[0][key]  # Keep single fps value
      else:
        # Stack along new batch dimension: (num_motions, num_frames, ...)
        batch_log[key] = np.stack([log[key] for log in all_logs], axis=0)  # type: ignore[arg-type]

    log = batch_log
    print(
      f"Batch log shape: {dict((k, v.shape) for k, v in log.items() if isinstance(v, np.ndarray))}"
    )
  else:
    # Single motion processing
    should_render = render
    log, frames = process_single_motion_sim(
      sim=sim,
      scene=scene,
      motion=motion,
      robot=robot,
      robot_joint_indexes=robot_joint_indexes,
      object_entity=object_entity,
      has_object_in_scene=has_object_in_scene,
      eef_indexes=eef_indexes,
      render=should_render,
      renderer=renderer,
      contact_threshold=contact_threshold,
      output_fps=output_fps,
      task_name=task_name,
      contact_sensors=contact_sensors,
    )

  # Save and upload
  print("\nSaving to /tmp/motion.npz...")
  np.savez("/tmp/motion.npz", **log)  # type: ignore[arg-type]

  print("Uploading to Weights & Biases...")
  import wandb

  collection_suffix = project_name

  COLLECTION = f"{collection_name}_{collection_suffix}"
  run = wandb.init(project=project_name, name=COLLECTION)
  print(f"[INFO]: Logging motion to wandb: {COLLECTION}")
  REGISTRY = "motions"
  logged_artifact = run.log_artifact(
    artifact_or_path="/tmp/motion.npz", name=COLLECTION, type=REGISTRY
  )
  run.link_artifact(
    artifact=logged_artifact,
    target_path=f"wandb-registry-{REGISTRY}/{COLLECTION}",
  )
  print(f"[INFO]: Motion saved to wandb registry: {REGISTRY}/{COLLECTION}")

  if render and len(frames) > 0:
    from moviepy import ImageSequenceClip

    print("Creating video...")
    clip = ImageSequenceClip(frames, fps=output_fps)
    clip.write_videofile("./motion.mp4")

    print("Logging video to wandb...")
    wandb.log({"motion_video": wandb.Video("./motion.mp4", format="mp4")})

  wandb.finish()


def main(
  task_name: str,
  input_file: str,
  project_name: str,
  collection_name: str | None = None,
  input_fps: float = 100.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
):
  """Replay motion from CSV file and output to npz file.

  Args:
    task_name: Task name from the registry (e.g., "Mjlab-Tracking-Flat-Unitree-G1-LargeBox").
    input_file: Path to the input CSV file.
    project_name: Wandb project name.
    collection_name: Collection name for the artifact. If None, extracted from input_file basename.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    render: Whether to render the simulation and save a video.
    line_range: Range of lines to process from the CSV file.
  """
  # Import tasks to populate the registry
  import mjlab.tasks  # noqa: F401

  # Extract collection_name from input_file if not provided
  if collection_name is None:
    # Get parent directory name (e.g., "motions/dir-name/my_motion.csv" -> "dir-name")
    collection_name = os.path.basename(os.path.dirname(input_file))
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  # Load environment config from task name
  env_cfg = load_env_cfg(task_name, play=False)
  scene = Scene(env_cfg.scene, device=device)
  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  scene.initialize(sim.mj_model, sim.model, sim.data)

  # Extract contact sensors from env_cfg in the expected order
  # Order: left_eef, right_eef, left_foot, right_foot (matching EE_SITE_NAMES)
  expected_sensor_names = [
    "left_eef_contact",
    "right_eef_contact",
    "left_foot_contact",
    "right_foot_contact",
  ]
  contact_sensors: dict[str, ContactSensorCfg] = {}
  for sensor_name in expected_sensor_names:
    for sensor_cfg in env_cfg.scene.sensors:
      if isinstance(sensor_cfg, ContactSensorCfg) and sensor_cfg.name == sensor_name:
        contact_sensors[sensor_name] = sensor_cfg
        break

  renderer = None
  if render:
    viewer_cfg = ViewerConfig(
      height=480,
      width=640,
      origin_type=ViewerConfig.OriginType.ASSET_ROOT,
      asset_name="robot",  # Specify robot entity for camera tracking when multiple entities exist
      distance=2.0,
      elevation=-5.0,
      azimuth=20,
    )
    renderer = OffscreenRenderer(
      model=sim.mj_model,
      cfg=viewer_cfg,
      scene=scene,
    )
    renderer.initialize()

  run_sim(
    sim=sim,
    scene=scene,
    joint_names=[
      "left_hip_pitch_joint",
      "left_hip_roll_joint",
      "left_hip_yaw_joint",
      "left_knee_joint",
      "left_ankle_pitch_joint",
      "left_ankle_roll_joint",
      "right_hip_pitch_joint",
      "right_hip_roll_joint",
      "right_hip_yaw_joint",
      "right_knee_joint",
      "right_ankle_pitch_joint",
      "right_ankle_roll_joint",
      "waist_yaw_joint",
      "waist_roll_joint",
      "waist_pitch_joint",
      "left_shoulder_pitch_joint",
      "left_shoulder_roll_joint",
      "left_shoulder_yaw_joint",
      "left_elbow_joint",
      "left_wrist_roll_joint",
      "left_wrist_pitch_joint",
      "left_wrist_yaw_joint",
      "right_shoulder_pitch_joint",
      "right_shoulder_roll_joint",
      "right_shoulder_yaw_joint",
      "right_elbow_joint",
      "right_wrist_roll_joint",
      "right_wrist_pitch_joint",
      "right_wrist_yaw_joint",
    ],
    input_fps=input_fps,
    input_file=input_file,
    output_fps=output_fps,
    project_name=project_name,
    collection_name=collection_name,
    render=render,
    line_range=line_range,
    renderer=renderer,
    contact_threshold=CONTACT_THRESHOLD,
    task_name=task_name,
    contact_sensors=contact_sensors if contact_sensors else None,
  )


if __name__ == "__main__":
  tyro.cli(main)
