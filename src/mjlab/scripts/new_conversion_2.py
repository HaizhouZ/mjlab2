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
  unitree_g1_flat_tracking_env_cfg_largebox,
)
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

CONTACT_THRESHOLD = 0.05  # 5cm
EE_SITE_NAMES = ["left_palm", "right_palm", "left_foot_tip", "right_foot_tip"]


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
        if mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
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
        if mj_model.geom_type[geom_id] == mujoco.mjtGeom.mjGEOM_BOX:
          object_size = mj_model.geom_size[geom_id].copy()
    except Exception:
      pass

  return object_size


def extract_contacts_from_mujoco(
  mj_model: mujoco.MjModel,
  mj_data: mujoco.MjData,
  robot: Entity,
  object_entity: Entity | None,
  eef_site_names: list[str],
) -> dict[str, Any]:
  """Extract contact information from MuJoCo simulation data.

  Uses MuJoCo's physics engine to detect actual contacts between end effector geoms
  and object geoms.

  Args:
    mj_model: MuJoCo model.
    mj_data: MuJoCo data containing contact information.
    robot: Robot entity.
    object_entity: Object entity, or None if no object.
    eef_site_names: List of end effector site names to check for contacts.

  Returns:
    Dictionary with contact information:
      - contact_positions: (num_eefs, 3) array of contact positions (NaN if no contact)
      - contact_indicators: (num_eefs,) boolean array indicating if contact exists
  """
  num_eefs = len(eef_site_names)
  contact_indicators = np.zeros(num_eefs, dtype=bool)
  contact_positions = np.full((num_eefs, 3), np.nan, dtype=np.float32)

  if object_entity is None:
    return {
      "contact_positions": contact_positions,
      "contact_indicators": contact_indicators,
    }

  # Get geom IDs for end effectors (from sites)
  eef_site_indices, _ = robot.find_sites(eef_site_names, preserve_order=True)
  eef_geom_ids = []
  for site_idx in eef_site_indices:
    if site_idx >= 0:
      # Get body ID from site
      site_body_id = int(mj_model.site_bodyid[site_idx])
      # Find geoms attached to this body
      body_geom_ids = []
      for geom_id in range(mj_model.ngeom):
        if int(mj_model.geom_bodyid[geom_id]) == site_body_id:
          body_geom_ids.append(geom_id)
      # Use first geom if found, otherwise use -1
      eef_geom_ids.append(body_geom_ids[0] if body_geom_ids else -1)
    else:
      eef_geom_ids.append(-1)

  # Get geom IDs for object
  object_geom_ids = []
  if hasattr(object_entity, "indexing") and hasattr(object_entity.indexing, "geom_ids"):
    geom_ids = object_entity.indexing.geom_ids
    for geom_id in geom_ids:
      geom_id_val = geom_id.item() if hasattr(geom_id, "item") else int(geom_id)
      object_geom_ids.append(int(geom_id_val))
  else:
    # Fallback: try to find object geoms by body name
    object_body_names = ["largebox_link", "box", "object"]
    for body_name in object_body_names:
      body_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, body_name)
      if body_id >= 0:
        for geom_id in range(mj_model.ngeom):
          if int(mj_model.geom_bodyid[geom_id]) == body_id:
            object_geom_ids.append(geom_id)
        break

  if not object_geom_ids:
    # No object geoms found, return all zeros/NaNs
    return {
      "contact_positions": contact_positions,
      "contact_indicators": contact_indicators,
    }

  # Track best contact for each end effector (smallest distance = most penetration)
  best_contact_dist = np.full(num_eefs, np.inf, dtype=np.float32)

  # Iterate through all contacts and find matches
  for contact_idx in range(mj_data.ncon):
    contact = mj_data.contact[contact_idx]
    geom1_id = int(contact.geom[0])
    geom2_id = int(contact.geom[1])

    # Check if this contact involves an end effector and the object
    for eef_idx, eef_geom_id in enumerate(eef_geom_ids):
      if eef_geom_id < 0:
        continue

      # Check if contact is between this end effector and any object geom
      if (geom1_id == eef_geom_id and geom2_id in object_geom_ids) or (
        geom2_id == eef_geom_id and geom1_id in object_geom_ids
      ):
        # Contact found!
        contact_indicators[eef_idx] = True

        # Keep the contact with smallest distance (most penetration)
        # Negative distance means penetration, so smaller = more penetration
        if contact.dist < best_contact_dist[eef_idx]:
          best_contact_dist[eef_idx] = contact.dist
          contact_positions[eef_idx] = contact.pos.copy()

  return {
    "contact_positions": contact_positions,
    "contact_indicators": contact_indicators,
  }


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


class MotionLoader:
  def __init__(
    self,
    motion_file: str,
    input_fps: int,
    output_fps: int,
    device: torch.device | str,
    line_range: tuple[int, int] | None = None,
  ):
    self.motion_file = motion_file
    self.input_fps = input_fps
    self.output_fps = output_fps
    self.input_dt = 1.0 / self.input_fps
    self.output_dt = 1.0 / self.output_fps
    self.current_idx = 0
    self.device = device
    self.line_range = line_range
    self._load_motion()
    self._interpolate_motion()
    self._compute_velocities()

  def _load_motion(self):
    """Loads the motion from the csv file."""
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
    print(
      f"[Loader] CSV columns: {num_cols}, Joints: {num_joints}, PD targets: yes, Object: yes"
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


def run_sim(
  sim: Simulation,
  scene: Scene,
  joint_names,
  input_file,
  input_fps,
  output_fps,
  output_name,
  render,
  line_range,
  renderer: OffscreenRenderer | None = None,
  contact_threshold: float = 0.05,
):
  motion = MotionLoader(
    motion_file=input_file,
    input_fps=input_fps,
    output_fps=output_fps,
    device=sim.device,
    line_range=line_range,
  )

  robot: Entity = scene["robot"]
  robot_joint_indexes = robot.find_joints(joint_names, preserve_order=True)[0]

  # Try to get object entity if it exists
  try:
    object_entity: Entity | None = scene.entities.get("box")
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
    # Contact information
    log["contact_positions"] = []  # (T, num_eefs, 3)
    log["contact_indicators"] = []  # (T, num_eefs)

  file_saved = False

  frames = []
  scene.reset()

  print(f"\nStarting simulation with {motion.output_frames} frames...")
  if render:
    print("Rendering enabled - generating video frames...")

  # Create progress bar
  pbar = tqdm(
    total=motion.output_frames,
    desc="Processing frames",
    unit="frame",
    ncols=100,
    bar_format="{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]",
  )

  frame_count = 0
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

    # Set object state if available
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

      # Log PD targets
      log["joint_pd_targets"].append(motion_pd_targets[0].cpu().numpy().copy())
      print(f"PD targets: {motion_pd_targets[0]}")

      # Log object states if available
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

          # Extract contact information
          # Get end effector positions from site_pos_w
          site_pos_w = robot.data.site_pos_w[0, :].cpu().numpy().copy()
          eef_positions = np.array(
            [
              site_pos_w[idx] if idx >= 0 else np.array([np.nan, np.nan, np.nan])
              for idx in eef_indexes
            ]
          )

          # Get object size from MuJoCo model (half-extents for box geometry)
          object_size = get_object_size(sim.mj_model, object_entity)

          # if object_size is not None:
          #   print(f"[Contact Detection] Object size (half-extents): {object_size}")
          #   print(
          #     f"[Contact Detection] Effective bounding box: ±{object_size + contact_threshold} (size + threshold)"
          #   )
          # else:
          #   print(
          #     f"[Contact Detection] Warning: Object size not found, using simple distance threshold ({contact_threshold}m) from center"
          #   )

          contact_data = extract_contacts_from_distance(
            eef_positions,
            curr_obj_pos,
            object_rotation=curr_obj_rot,
            object_size=object_size,
            threshold=contact_threshold,
          )

          log["contact_positions"].append(contact_data["contact_positions"])  # type: ignore[union-attr]
          log["contact_indicators"].append(contact_data["contact_indicators"])  # type: ignore[union-attr]
        else:
          # Log motion data directly if object entity doesn't exist in scene
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

      if frame_count % 100 == 0:  # Update every 100 frames to avoid spam
        elapsed_time = frame_count / output_fps
        pbar.set_description(f"Processing frames (t={elapsed_time:.1f}s)")

      if reset_flag and not file_saved:
        file_saved = True
        pbar.close()

        print("\nStacking arrays and saving data...")
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
          # Stack contact information
          if len(log["contact_positions"]) > 0:  # type: ignore[arg-type]
            log["contact_positions"] = np.stack(
              log["contact_positions"],
              axis=0,  # type: ignore[arg-type]
            )  # (T, num_eefs, 3)
            log["contact_indicators"] = np.stack(
              log["contact_indicators"],
              axis=0,  # type: ignore[arg-type]
            )  # (T, num_eefs)

        print("Saving to /tmp/motion.npz...")
        np.savez("/tmp/motion.npz", **log)  # type: ignore[arg-type]

        print("Uploading to Weights & Biases...")
        import wandb

        COLLECTION = output_name
        run = wandb.init(project="csv_to_npz", name=COLLECTION)
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

        if render:
          from moviepy import ImageSequenceClip

          print("Creating video...")
          clip = ImageSequenceClip(frames, fps=output_fps)
          clip.write_videofile("./motion.mp4")

          print("Logging video to wandb...")
          wandb.log({"motion_video": wandb.Video("./motion.mp4", format="mp4")})

        wandb.finish()


def main(
  input_file: str,
  output_name: str,
  input_fps: float = 100.0,
  output_fps: float = 50.0,
  device: str = "cuda:0",
  render: bool = False,
  line_range: tuple[int, int] | None = None,
):
  """Replay motion from CSV file and output to npz file.

  Args:
    input_file: Path to the input CSV file.
    output_name: Path to the output npz file.
    input_fps: Frame rate of the CSV file.
    output_fps: Desired output frame rate.
    device: Device to use.
    render: Whether to render the simulation and save a video.
    line_range: Range of lines to process from the CSV file.
  """
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = 1.0 / output_fps

  scene = Scene(unitree_g1_flat_tracking_env_cfg_largebox().scene, device=device)
  model = scene.compile()

  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)

  scene.initialize(sim.mj_model, sim.model, sim.data)

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
    output_name=output_name,
    render=render,
    line_range=line_range,
    renderer=renderer,
    contact_threshold=CONTACT_THRESHOLD,
  )


if __name__ == "__main__":
  tyro.cli(main)
