"""Convert NPZ motion files (SBTO or OmniRetarget format) to CSV format."""

import pickle
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation, Slerp

from mjlab.utils import run_cli_function


def quat_mul_wxyz(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
  """Multiply two quaternions in wxyz format.

  Args:
    q1: First quaternion in (w, x, y, z) format. Shape (..., 4) or (4,)
    q2: Second quaternion in (w, x, y, z) format. Shape (..., 4) or (4,)

  Returns:
    Product quaternion in (w, x, y, z) format. Same shape as inputs.
  """
  # Ensure 2D for easier handling
  was_1d = False
  if q1.ndim == 1:
    q1 = q1.reshape(1, -1)
    was_1d = True
  if q2.ndim == 1:
    q2 = q2.reshape(1, -1)

  # Extract components (wxyz format)
  w1, x1, y1, z1 = q1[:, 0], q1[:, 1], q1[:, 2], q1[:, 3]
  w2, x2, y2, z2 = q2[:, 0], q2[:, 1], q2[:, 2], q2[:, 3]

  # Quaternion multiplication formula for wxyz format
  w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
  x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
  y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
  z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

  result = np.stack([w, x, y, z], axis=-1)

  if was_1d:
    result = result.reshape(-1)

  return result


def normalize_quat_wxyz(q: np.ndarray) -> np.ndarray:
  """Normalize quaternion in wxyz format.

  Args:
    q: Quaternion in (w, x, y, z) format. Shape (..., 4) or (4,)

  Returns:
    Normalized quaternion in (w, x, y, z) format.
  """
  norm = np.linalg.norm(q, axis=-1, keepdims=True)
  if q.ndim == 1:
    norm = norm.reshape(-1)
  return q / np.maximum(norm, 1e-8)


def ease_out_cubic(t):
  """Cubic ease-out curve for smooth deceleration."""
  return 1 - (1 - t) ** 3


def ease_in_cubic(t):
  """Cubic ease-in curve for smooth acceleration."""
  return t**3


def process_single_motion(
  frames: np.ndarray,
  num_joints: int,
  fps: float,
  duration: float | None,
  pad_duration: float,
  transition_duration: float,
  add_start_transition: bool,
  add_end_transition: bool,
) -> np.ndarray:
  """Process a single motion (2D array: num_frames x features).

  Args:
    frames: Motion frames array of shape (num_frames, features).
    num_joints: Number of joints.
    fps: Frames per second.
    duration: Desired duration in seconds. If None, use original duration.
    pad_duration: Duration in seconds to hold the final pose at the end.
    transition_duration: Duration in seconds to blend to/from safe standing pose.
    add_start_transition: Whether to add transition from safe standing pose to motion start.
    add_end_transition: Whether to add transition from motion end to safe standing pose.

  Returns:
    Processed frames array of shape (num_frames, features).
  """
  # Hardcoded safe standing pose joints.
  # fmt: off
  SAFE_POSE_JOINTS = np.array(
    [
      -0.312, 0, 0, 0.669, -0.363, 0,  # left leg
      -0.312, 0, 0, 0.669, -0.363, 0,  # right leg
      0, 0, 0,  # waist
      0.2, 0.2, 0, 0.6, 0, 0, 0,  # left arm
      0.2, -0.2, 0, 0.6, 0, 0, 0,  # right arm
    ]
  )
  # fmt: on

  SAFE_Z_HEIGHT = 0.76  # Safe standing height.

  original_duration = frames.shape[0] / fps

  # Repeat motion if duration is specified.
  if duration is not None:
    if duration < original_duration:
      print(
        f"Warning: Requested duration ({duration}s) is shorter than original ({original_duration:.2f}s)"
      )
      print("         Truncating motion...")
      num_frames = int(duration * fps)
      frames = frames[:num_frames]
    else:
      num_cycles = int(np.ceil(duration / original_duration))
      print(f"Repeating motion {num_cycles} times to reach {duration}s...")

      # Calculate displacement per cycle for all position components (base + object, XY only).
      start_frame = frames[0]
      end_frame = frames[-1]

      # Base position displacement (indices 0:3)
      base_displacement = end_frame[0:3] - start_frame[0:3]
      base_displacement[2] = 0.0  # Only accumulate XY

      # Object position displacement (indices 6+2*num_joints : 6+2*num_joints+3)
      obj_pos_start_idx = 6 + 2 * num_joints
      obj_pos_end_idx = obj_pos_start_idx + 3
      obj_displacement = (
        end_frame[obj_pos_start_idx:obj_pos_end_idx]
        - start_frame[obj_pos_start_idx:obj_pos_end_idx]
      )
      obj_displacement[2] = 0.0  # Only accumulate XY

      print(
        f"Base XY displacement per cycle: [{base_displacement[0]:.4f}, {base_displacement[1]:.4f}]"
      )
      print(
        f"Object XY displacement per cycle: [{obj_displacement[0]:.4f}, {obj_displacement[1]:.4f}]"
      )
      if np.linalg.norm(base_displacement[:2]) < 1e-3:
        print("  (In-place motion detected)")
      else:
        print("  (Forward motion detected - accumulating XY displacement only)")

      # Repeat and accumulate XY displacement only.
      repeated_frames = []
      for cycle in range(num_cycles):
        cycle_frames = frames.copy()
        # Add cumulative displacement to base position (XY only).
        cycle_frames[:, 0:3] += base_displacement * cycle
        # Add cumulative displacement to object position (XY only).
        cycle_frames[:, obj_pos_start_idx:obj_pos_end_idx] += obj_displacement * cycle
        repeated_frames.append(cycle_frames)

      frames = np.vstack(repeated_frames)

      # Truncate to exact duration.
      num_frames = int(duration * fps)
      frames = frames[:num_frames]

  # Add transition FROM safe standing pose TO motion start.
  if add_start_transition:
    print("\nAdding start transition from safe standing pose to motion start...")

    # Get the first frame as target.
    target_frame = frames[0].copy()

    # Number of transition frames.
    transition_frames = int(transition_duration * fps)
    print(
      f"Creating {transition_duration}s start transition ({transition_frames} frames)..."
    )

    # Extract components from target (first) frame.
    target_pos = target_frame[0:3]
    target_rot = target_frame[3:6]  # axis-angle
    target_joints = target_frame[6 : 6 + num_joints]
    target_pd_targets = target_frame[6 + num_joints : 6 + 2 * num_joints]
    # Extract object data
    obj_pos_start_idx = 6 + 2 * num_joints
    obj_pos_end_idx = obj_pos_start_idx + 3
    obj_rot_start_idx = obj_pos_end_idx
    obj_rot_end_idx = obj_rot_start_idx + 3
    target_obj_pos = target_frame[obj_pos_start_idx:obj_pos_end_idx]
    target_obj_rot = target_frame[obj_rot_start_idx:obj_rot_end_idx]  # axis-angle

    # Convert target rotation to euler angles to extract yaw.
    target_rot_obj = Rotation.from_rotvec(target_rot)
    euler_zyx = target_rot_obj.as_euler("ZYX", degrees=False)
    yaw = euler_zyx[0]

    print(
      f"  First frame orientation: yaw={np.degrees(yaw):.1f}°, pitch={np.degrees(euler_zyx[1]):.1f}°, roll={np.degrees(euler_zyx[2]):.1f}°"
    )
    print(
      f"  Start orientation: yaw={np.degrees(yaw):.1f}° (matched), pitch=0°, roll=0°"
    )

    # Create start rotation: yaw from first frame, roll=0, pitch=0.
    start_rot_obj = Rotation.from_euler("ZYX", [yaw, 0, 0], degrees=False)
    start_rot = start_rot_obj.as_rotvec()

    # Start position: XY from first frame, Z at safe height.
    start_pos = target_pos.copy()
    start_pos[2] = SAFE_Z_HEIGHT

    # Start joints: safe standing pose.
    start_joints = SAFE_POSE_JOINTS
    # Start PD targets: same as joints (use safe pose as PD target)
    start_pd_targets = SAFE_POSE_JOINTS

    # Object transition setup - use first frame position and rotation as start
    start_obj_pos = target_obj_pos.copy()
    start_obj_rot_obj = Rotation.from_rotvec(target_obj_rot)
    target_obj_rot_obj = Rotation.from_rotvec(target_obj_rot)
    key_times_obj = [0, 1]
    key_rots_obj = Rotation.concatenate([start_obj_rot_obj, target_obj_rot_obj])
    obj_slerp = Slerp(key_times_obj, key_rots_obj)

    # Create Slerp interpolator for smooth rotation transition.
    key_times = [0, 1]
    key_rots = Rotation.concatenate([start_rot_obj, target_rot_obj])
    slerp = Slerp(key_times, key_rots)

    # Create transition frames.
    start_transition = []
    for i in range(transition_frames):
      t = i / (transition_frames - 1) if transition_frames > 1 else 1.0
      t_eased = ease_in_cubic(t)  # Apply ease-in for acceleration.

      # LERP position.
      pos = start_pos * (1 - t_eased) + target_pos * t_eased

      # SLERP rotation.
      rot_obj = slerp(t_eased)
      rot = rot_obj.as_rotvec()

      # LERP joints.
      joints = start_joints * (1 - t_eased) + target_joints * t_eased

      # LERP PD targets.
      pd_targets_frame = start_pd_targets * (1 - t_eased) + target_pd_targets * t_eased

      # Object interpolation
      obj_pos = start_obj_pos * (1 - t_eased) + target_obj_pos * t_eased
      obj_rot_obj = obj_slerp(t_eased)
      obj_rot = obj_rot_obj.as_rotvec()

      # Combine.
      frame = np.concatenate([pos, rot, joints, pd_targets_frame, obj_pos, obj_rot])
      start_transition.append(frame)

    start_transition = np.array(start_transition)
    frames = np.vstack([start_transition, frames])

  # Add transition TO safe standing pose at the end.
  if add_end_transition:
    print(
      "\nAdding end transition to safe standing pose (keeping yaw, resetting roll/pitch)..."
    )

    # Get the last frame as starting point for transition.
    start_frame = frames[-1].copy()

    # Number of transition frames.
    transition_frames = int(transition_duration * fps)
    print(
      f"Creating {transition_duration}s end transition ({transition_frames} frames)..."
    )

    # Extract components from start frame.
    start_pos = start_frame[0:3]
    start_rot = start_frame[3:6]  # axis-angle
    start_joints = start_frame[6 : 6 + num_joints]
    start_pd_targets = start_frame[6 + num_joints : 6 + 2 * num_joints]
    # Extract object data
    obj_pos_start_idx = 6 + 2 * num_joints
    obj_pos_end_idx = obj_pos_start_idx + 3
    obj_rot_start_idx = obj_pos_end_idx
    obj_rot_end_idx = obj_rot_start_idx + 3
    start_obj_pos = start_frame[obj_pos_start_idx:obj_pos_end_idx]
    start_obj_rot = start_frame[obj_rot_start_idx:obj_rot_end_idx]  # axis-angle

    # Convert start rotation to euler angles to extract yaw.
    start_rot_obj = Rotation.from_rotvec(start_rot)
    euler_zyx = start_rot_obj.as_euler("ZYX", degrees=False)
    yaw = euler_zyx[0]  # Extract yaw.

    print(
      f"  Final frame orientation: yaw={np.degrees(yaw):.1f}°, pitch={np.degrees(euler_zyx[1]):.1f}°, roll={np.degrees(euler_zyx[2]):.1f}°"
    )
    print(f"  Target orientation: yaw={np.degrees(yaw):.1f}° (kept), pitch=0°, roll=0°")

    # Create target rotation: yaw from final frame, roll=0, pitch=0.
    target_rot_obj = Rotation.from_euler("ZYX", [yaw, 0, 0], degrees=False)
    target_rot = target_rot_obj.as_rotvec()

    # Target: keep XY position, transition Z to safe height.
    target_pos = start_pos.copy()
    target_pos[2] = SAFE_Z_HEIGHT

    # Target joints from safe pose.
    target_joints = SAFE_POSE_JOINTS
    # Target PD targets: same as joints (use safe pose as PD target)
    target_pd_targets = SAFE_POSE_JOINTS

    # Object transition setup - keep final frame position and rotation as target
    target_obj_pos = start_obj_pos.copy()
    start_obj_rot_obj_end = Rotation.from_rotvec(start_obj_rot)
    target_obj_rot_obj_end = Rotation.from_rotvec(start_obj_rot)  # Keep same rotation
    key_times_obj = [0, 1]
    key_rots_obj = Rotation.concatenate([start_obj_rot_obj_end, target_obj_rot_obj_end])
    obj_slerp = Slerp(key_times_obj, key_rots_obj)

    # Create Slerp interpolator for smooth rotation transition.
    key_times = [0, 1]
    key_rots = Rotation.concatenate([start_rot_obj, target_rot_obj])
    slerp = Slerp(key_times, key_rots)

    # Create transition frames.
    end_transition = []
    for i in range(transition_frames):
      t = i / (transition_frames - 1) if transition_frames > 1 else 1.0
      t_eased = ease_out_cubic(t)  # Apply ease-out for deceleration.

      # LERP position.
      pos = start_pos * (1 - t_eased) + target_pos * t_eased

      # SLERP rotation.
      rot_obj = slerp(t_eased)
      rot = rot_obj.as_rotvec()

      # LERP joints.
      joints = start_joints * (1 - t_eased) + target_joints * t_eased

      # LERP PD targets.
      pd_targets_frame = start_pd_targets * (1 - t_eased) + target_pd_targets * t_eased

      # Object interpolation
      obj_pos = start_obj_pos * (1 - t_eased) + target_obj_pos * t_eased
      obj_rot_obj = obj_slerp(t_eased)
      obj_rot = obj_rot_obj.as_rotvec()

      # Combine.
      frame = np.concatenate([pos, rot, joints, pd_targets_frame, obj_pos, obj_rot])
      end_transition.append(frame)

    end_transition = np.array(end_transition)
    frames = np.vstack([frames, end_transition])

  # Add padding at the end (hold final pose).
  if pad_duration > 0:
    pad_frames = int(pad_duration * fps)
    print(
      f"\nAdding {pad_duration}s padding ({pad_frames} frames) - holding final pose..."
    )

    # Repeat the last frame.
    final_frame = frames[-1:].copy()
    padding = np.tile(final_frame, (pad_frames, 1))

    frames = np.vstack([frames, padding])

  return frames


def reverse_motion_frames(frames: np.ndarray) -> np.ndarray:
  """Reverse motion frames along the time dimension.

  Args:
    frames: Motion frames array of shape (num_frames, features).

  Returns:
    Reversed frames array of shape (num_frames, features).
  """
  # Reverse along the time dimension (axis=0)
  reversed_frames = np.flip(frames, axis=0)
  return reversed_frames


def transform_to_first_frame_origin(frames: np.ndarray, num_joints: int) -> np.ndarray:
  """Transform frames so that the robot's first frame starts at (x, y) = 0 and yaw = 0.

  This applies a coordinate frame transformation (not normalization) where:
  - All positions are translated so the first frame's (x, y) is at origin
  - All positions and orientations are rotated so the first frame's yaw is 0

  Args:
    frames: Motion frames array of shape (num_frames, features).
      Format: [root_pos(3), axis_angle(3), joints(num_joints), joint_pd_targets(num_joints),
               obj_pos(3), obj_axis_angle(3)]
    num_joints: Number of joints.

  Returns:
    Transformed frames array with same shape as input.
  """
  if frames.shape[0] == 0:
    return frames

  # Extract first frame's position and rotation
  first_frame = frames[0]
  first_pos = first_frame[0:3]  # (x, y, z)
  first_rot_axis_angle = first_frame[3:6]  # axis-angle

  # Convert first frame's rotation to euler angles to extract yaw
  first_rot_obj = Rotation.from_rotvec(first_rot_axis_angle)
  euler_zyx = first_rot_obj.as_euler("ZYX", degrees=False)
  first_yaw = euler_zyx[0]  # Extract yaw (rotation around z-axis)

  # Create inverse transformation:
  # 1. Translation: subtract first frame's (x, y) position
  # 2. Rotation: rotate by -first_yaw around z-axis

  # Create rotation matrix for -first_yaw around z-axis
  inverse_yaw_rot = Rotation.from_euler("Z", -first_yaw, degrees=False)

  # Extract components from all frames
  root_pos = frames[:, 0:3].copy()  # (N, 3)
  root_rot_axis_angle = frames[:, 3:6].copy()  # (N, 3)
  joint_dof = frames[:, 6 : 6 + num_joints].copy()  # (N, num_joints)
  joint_pd_targets = frames[
    :, 6 + num_joints : 6 + 2 * num_joints
  ].copy()  # (N, num_joints)
  obj_pos_start_idx = 6 + 2 * num_joints
  obj_pos_end_idx = obj_pos_start_idx + 3
  obj_rot_start_idx = obj_pos_end_idx
  obj_rot_end_idx = obj_rot_start_idx + 3
  obj_pos = frames[:, obj_pos_start_idx:obj_pos_end_idx].copy()  # (N, 3)
  obj_rot_axis_angle = frames[:, obj_rot_start_idx:obj_rot_end_idx].copy()  # (N, 3)

  # Transform root position:
  # Order: First translate, then rotate around origin
  # 1. Translate to make first frame position the origin for rotation
  root_pos[:, 0:2] -= first_pos[0:2]
  # 2. Rotate by -first_yaw around z-axis (around the translated origin)
  root_pos = inverse_yaw_rot.apply(root_pos)

  # Transform root rotation:
  # Vectorized: convert all rotations at once, compose, convert back
  root_rotations = Rotation.from_rotvec(root_rot_axis_angle)  # (N,) Rotation object
  root_rotations_transformed = (
    inverse_yaw_rot * root_rotations
  )  # Vectorized composition
  root_rot_axis_angle = root_rotations_transformed.as_rotvec()  # (N, 3)

  # Transform object position:
  # Same transformation as root position (both in world coordinates)
  # 1. Translate to make first frame position the origin for rotation
  obj_pos[:, 0:2] -= first_pos[0:2]
  # 2. Rotate by -first_yaw around z-axis (around the translated origin)
  obj_pos = inverse_yaw_rot.apply(obj_pos)

  # Transform object rotation:
  # Vectorized: convert all rotations at once, compose, convert back
  obj_rotations = Rotation.from_rotvec(obj_rot_axis_angle)  # (N,) Rotation object
  obj_rotations_transformed = inverse_yaw_rot * obj_rotations  # Vectorized composition
  obj_rot_axis_angle = obj_rotations_transformed.as_rotvec()  # (N, 3)

  # Reconstruct frames
  transformed_frames = np.concatenate(
    [
      root_pos,
      root_rot_axis_angle,
      joint_dof,
      joint_pd_targets,
      obj_pos,
      obj_rot_axis_angle,
    ],
    axis=1,
  )

  return transformed_frames


def main(
  input_file: str,
  csv_file: str,
  duration: float | None = None,
  pad_duration: float = 0.0,
  transition_duration: float = 1.0,
  add_start_transition: bool = False,
  add_end_transition: bool = False,
  add_reverse: bool = False,
):
  """Convert NPZ/PKL motion file (SBTO, OmniRetarget, Ilyass pickle, or Ilyass npz format) to CSV format.

  Output format: [root_pos(3), quat_xyzw(4), joint_dof(N), joint_pd_targets(N), obj_pos(3), obj_quat_xyzw(4)]

  Supports four input formats:
  - SBTO format: requires 'base_xyz_quat', 'actuator_pos', 'obj_0_xyz_quat', and 'u' keys
  - OmniRetarget format: requires 'qpos' key with shape (T, 43) or (M, T, 43)
    - qpos[:, 0:4] = base quat (wxyz)
    - qpos[:, 4:7] = base pos (xyz)
    - qpos[:, 7:36] = joint positions (29D)
    - qpos[:, 36:40] = object quat (wxyz)
    - qpos[:, 40:43] = object pos (xyz)
  - Ilyass pickle format: requires 'fps', 'root_pos', 'root_rot', 'dof_pos', 'object_pos', 'object_rot' keys
    - root_pos: (T, 3) base position
    - root_rot: (T, 4) base rotation in xyzw format
    - dof_pos: (T, 29) joint positions
    - object_pos: (T, 3) object position
    - object_rot: (T, 4) object rotation in xyzw format
  - Ilyass npz format: same as Ilyass pickle format but stored in .npz file instead of .pkl

  Supports both single motion (num_frames, ...) and batch of motions (num_motions, num_frames, ...).
  If input is 3D, output will be saved as NPZ to preserve shape.

  Args:
    input_file: Path to input .npz or .pkl file (SBTO, OmniRetarget, Ilyass pickle, or Ilyass npz format).
    csv_file: Path to output .csv file (or .npz if input is 3D).
    duration: Desired duration in seconds. If None, use original duration. Motion will
      be cycled to reach this duration.
    pad_duration: Duration in seconds to hold the final pose at the end.
    transition_duration: Duration in seconds to blend to/from safe standing pose.
    add_start_transition: Whether to add transition from safe standing pose to motion start.
    add_end_transition: Whether to add transition from motion end to safe standing pose.
    add_reverse: Whether to append the reversed motion trajectory at the end.
  """

  OBJECT_POS_OFFSET = np.array(
    [0.0, 0.0, 0.0105]
  )  # from omniretarget largebox mesh (xyz)
  OBJECT_ROT_OFFSET = np.array(
    [0.00991298, 0.849052, -0.523456, 0.0707591]
  )  # from omniretarget largebox mesh (wxyz)

  print(f"Loading {input_file}...")

  # Detect file format and load accordingly
  is_pickle_format = Path(input_file).suffix.lower() == ".pkl"

  if is_pickle_format:
    print("[Loader] Detected pickle format.")
    with open(input_file, "rb") as f:
      data = pickle.load(f)
    # Convert pickle dict to dict-like object compatible with existing code
    # The pickle format has keys: fps, root_pos, root_rot, dof_pos, object_pos, object_rot
    if not isinstance(data, dict):
      raise ValueError(
        f"Expected pickle file to contain a dictionary, got {type(data)}"
      )
  else:
    data = np.load(input_file, allow_pickle=True)
    print("[Loader] Detected NPZ format.")

  # Detect format: SBTO, OmniRetarget, Ilyass pickle, or Ilyass npz
  is_sbto_format = "base_xyz_quat" in data and "actuator_pos" in data
  is_omniretarget_format = "qpos" in data
  is_ilyass_pickle_format = (
    is_pickle_format
    and "fps" in data
    and "root_pos" in data
    and "root_rot" in data
    and "dof_pos" in data
    and "object_pos" in data
    and "object_rot" in data
  )
  is_ilyass_npz_format = (
    not is_pickle_format
    and "fps" in data
    and "root_pos" in data
    and "root_rot" in data
    and "dof_pos" in data
    and "object_pos" in data
    and "object_rot" in data
  )

  if (
    not is_sbto_format
    and not is_omniretarget_format
    and not is_ilyass_pickle_format
    and not is_ilyass_npz_format
  ):
    raise ValueError(
      f"Expected SBTO format (with 'base_xyz_quat' and 'actuator_pos' keys), "
      f"OmniRetarget format (with 'qpos' key), "
      f"Ilyass pickle format (with 'fps', 'root_pos', 'root_rot', 'dof_pos', 'object_pos', 'object_rot' keys), or "
      f"Ilyass npz format (with 'fps', 'root_pos', 'root_rot', 'dof_pos', 'object_pos', 'object_rot' keys). "
      f"Found keys: {list(data.keys())}"
    )

  # Detect if data is 3D (batch of motions) or 2D (single motion)
  num_motions = 0  # Initialize for type checking
  if is_sbto_format:
    base_xyz_quat = data["base_xyz_quat"].astype(np.float32)
    is_batch = base_xyz_quat.ndim == 3
    if is_batch:
      num_motions, num_frames_per_motion = base_xyz_quat.shape[:2]
      print(
        f"[Loader] Detected batch format: {num_motions} motions, {num_frames_per_motion} frames each"
      )
    else:
      print(f"[Loader] Detected single motion format: {base_xyz_quat.shape[0]} frames")
  elif is_ilyass_pickle_format or is_ilyass_npz_format:
    # Ilyass pickle/npz format is always single motion (no batch support)
    root_pos = np.asarray(data["root_pos"], dtype=np.float32)
    is_batch = False
    print(f"[Loader] Detected single motion format: {root_pos.shape[0]} frames")
  else:  # OmniRetarget
    qpos_np = data["qpos"].astype(np.float32)
    is_batch = qpos_np.ndim == 3
    if is_batch:
      num_motions, num_frames_per_motion = qpos_np.shape[:2]
      print(
        f"[Loader] Detected batch format: {num_motions} motions, {num_frames_per_motion} frames each"
      )
    else:
      print(f"[Loader] Detected single motion format: {qpos_np.shape[0]} frames")

  def load_single_motion(
    data: dict[str, Any], motion_idx: int | None = None
  ) -> tuple[np.ndarray, int, float]:
    """Load a single motion from data dictionary.

    Args:
      data: NPZ data dictionary or pickle data dictionary.
      motion_idx: Index of motion to load (for batch data). If None, loads single motion.

    Returns:
      Tuple of (frames, num_joints, fps) where frames is (num_frames, features).
    """
    if is_sbto_format:
      print("[Loader] Using SBTO format loader.")
      base_xyz_quat = data["base_xyz_quat"].astype(np.float32)
      actuator_pos = data["actuator_pos"].astype(np.float32)

      if motion_idx is not None:
        base_xyz_quat = base_xyz_quat[motion_idx]
        actuator_pos = actuator_pos[motion_idx]

      T = base_xyz_quat.shape[0]

      # Object data is required
      assert "obj_0_xyz_quat" in data, (
        f"Object data 'obj_0_xyz_quat' is required but not found in NPZ file. "
        f"Found keys: {list[Any](data.keys())}"
      )
      print("[Loader] Object data detected.")

      # Extract time array and infer FPS
      if "time" in data:
        times_np = data["time"].astype(np.float32)
        if motion_idx is not None and times_np.ndim > 1:
          times_np = times_np[motion_idx]
        times_np = np.asarray(times_np, dtype=np.float32).flatten()
        if T > 1:
          dt = float(np.mean(np.diff(times_np)))
          fps = 1.0 / dt if dt > 0 else 100.0
        else:
          fps = 100.0
      else:
        print("[Loader] Warning: 'time' key not found, assuming 100 FPS (dt=0.01s)")
        fps = 100.0

      # Extract root position
      root_pos = base_xyz_quat[:, 0:3]  # (N, 3)

      # Convert quaternion to axis-angle
      print("[Loader] Converting quaternions to axis-angle...")
      quats = base_xyz_quat[:, 3:7]  # (N, 4) - [qw, qx, qy, qz]
      quats = quats[:, [1, 2, 3, 0]]
      root_rot_axis_angle = []
      for quat in quats:
        # Normalize quaternion
        quat_norm = quat / (np.linalg.norm(quat) + 1e-8)
        rotation = Rotation.from_quat(quat_norm)
        rot_vec = rotation.as_rotvec()
        root_rot_axis_angle.append(rot_vec)
      root_rot_axis_angle = np.array(root_rot_axis_angle)  # (N, 3)

      # Extract joints
      joint_dof = actuator_pos  # (N, num_joints)
      num_joints = joint_dof.shape[1]

      # Extract PD targets from data["u"]
      assert "u" in data, (
        f"PD targets 'u' is required but not found in NPZ file. "
        f"Found keys: {list[Any](data.keys())}"
      )
      joint_pd_targets = data["u"].astype(
        np.float32
      )  # (N, num_joints) or (M, N, num_joints)
      if motion_idx is not None:
        joint_pd_targets = joint_pd_targets[motion_idx]
      assert joint_pd_targets.shape[1] == num_joints, (
        f"PD targets shape {joint_pd_targets.shape[1]} does not match number of joints {num_joints}"
      )
      # repeat the last frame of PD targets to match shape of joint_dof
      joint_pd_targets = np.concatenate(
        [joint_pd_targets, joint_pd_targets[None, -1]], axis=0
      )
      print("[Loader] PD targets detected.")

      # Extract object data
      obj = data["obj_0_xyz_quat"].astype(np.float32)
      if motion_idx is not None:
        obj = obj[motion_idx]
      object_pos = obj[:, 0:3]  # (N, 3)
      object_pos = object_pos - OBJECT_POS_OFFSET

      # Extract object quaternions (in wxyz format from NPZ: [qw, qx, qy, qz])
      obj_quats_wxyz = obj[:, 3:7]  # (N, 4) - [qw, qx, qy, qz] = wxyz format

      # Apply quaternion offset (OBJECT_ROT_OFFSET is in wxyz format)
      obj_rot_offset_wxyz = normalize_quat_wxyz(OBJECT_ROT_OFFSET)
      # Multiply each quaternion with the offset: result = q * offset (same as victor_npz_to_npz.py)
      obj_quats_wxyz_offset = np.array(
        [
          normalize_quat_wxyz(quat_mul_wxyz(obj_quats_wxyz[i], obj_rot_offset_wxyz))
          for i in range(obj_quats_wxyz.shape[0])
        ]
      )
      print("[Loader] Applying object pos & quaternion offset...")

      # Convert object quaternion to axis-angle
      print("[Loader] Converting object quaternions to axis-angle...")
      # Convert from wxyz to xyzw for scipy Rotation
      obj_quats_xyzw = obj_quats_wxyz_offset[:, [1, 2, 3, 0]]  # Convert wxyz -> xyzw

      object_rot_axis_angle = []
      for quat in obj_quats_xyzw:
        # Normalize quaternion
        quat_norm = quat / (np.linalg.norm(quat) + 1e-8)
        rotation = Rotation.from_quat(quat_norm)
        rot_vec = rotation.as_rotvec()
        object_rot_axis_angle.append(rot_vec)
      object_rot_axis_angle = np.array(object_rot_axis_angle)  # (N, 3)

    elif is_omniretarget_format:
      print("[Loader] Using OmniRetarget format loader.")
      qpos_np = data["qpos"].astype(np.float32)  # (T, 43) or (M, T, 43)

      if motion_idx is not None:
        qpos_np = qpos_np[motion_idx]

      T = qpos_np.shape[0]

      # Get fps if available, otherwise default to 30 fps
      if "fps" in data:
        fps = (
          int(data["fps"][0])
          if isinstance(data["fps"], np.ndarray)
          else int(data["fps"])
        )
      else:
        fps = 30
        print(f"[Loader] Warning: 'fps' not found, defaulting to {fps} fps")

      # Extract base pose: qpos[:, 0:7] = [qw, qx, qy, qz, x, y, z]
      base_quats_wxyz = qpos_np[:, 0:4]  # (T, 4) - wxyz format
      root_pos = qpos_np[:, 4:7]  # (T, 3)

      # Normalize base quaternions
      base_quats_wxyz = np.array([normalize_quat_wxyz(q) for q in base_quats_wxyz])

      # Convert base quaternion to axis-angle
      print("[Loader] Converting base quaternions to axis-angle...")
      # Convert from wxyz to xyzw for scipy Rotation
      base_quats_xyzw = base_quats_wxyz[:, [1, 2, 3, 0]]  # Convert wxyz -> xyzw
      root_rot_axis_angle = []
      for quat in base_quats_xyzw:
        # Normalize quaternion
        quat_norm = quat / (np.linalg.norm(quat) + 1e-8)
        rotation = Rotation.from_quat(quat_norm)
        rot_vec = rotation.as_rotvec()
        root_rot_axis_angle.append(rot_vec)
      root_rot_axis_angle = np.array(root_rot_axis_angle)  # (T, 3)

      # Extract joint positions: qpos[:, 7:36] (29D)
      joint_dof = qpos_np[:, 7:36]  # (T, 29)
      num_joints = joint_dof.shape[1]

      # OmniRetarget format doesn't have PD targets, set to zero
      joint_pd_targets = np.zeros_like(joint_dof)
      print("[Loader] PD targets not found in OmniRetarget format, setting to zero.")

      # Extract object pose: qpos[:, 36:43] = [qw, qx, qy, qz, x, y, z]
      obj_quats_wxyz = qpos_np[:, 36:40]  # (T, 4) - wxyz format
      object_pos = qpos_np[:, 40:43]  # (T, 3)

      # Normalize object quaternions
      obj_quats_wxyz = np.array([normalize_quat_wxyz(q) for q in obj_quats_wxyz])

      # Apply object position offset
      object_pos = object_pos - OBJECT_POS_OFFSET

      # Apply quaternion offset (OBJECT_ROT_OFFSET is in wxyz format)
      obj_rot_offset_wxyz = normalize_quat_wxyz(OBJECT_ROT_OFFSET)
      # Multiply each quaternion with the offset: result = q * offset
      obj_quats_wxyz_offset = np.array(
        [
          normalize_quat_wxyz(quat_mul_wxyz(obj_quats_wxyz[i], obj_rot_offset_wxyz))
          for i in range(obj_quats_wxyz.shape[0])
        ]
      )
      print("[Loader] Applying object pos & quaternion offset...")

      # Convert object quaternion to axis-angle
      print("[Loader] Converting object quaternions to axis-angle...")
      # Convert from wxyz to xyzw for scipy Rotation
      obj_quats_xyzw = obj_quats_wxyz_offset[:, [1, 2, 3, 0]]  # Convert wxyz -> xyzw

      object_rot_axis_angle = []
      for quat in obj_quats_xyzw:
        # Normalize quaternion
        quat_norm = quat / (np.linalg.norm(quat) + 1e-8)
        rotation = Rotation.from_quat(quat_norm)
        rot_vec = rotation.as_rotvec()
        object_rot_axis_angle.append(rot_vec)
      object_rot_axis_angle = np.array(object_rot_axis_angle)  # (T, 3)

    elif is_ilyass_pickle_format or is_ilyass_npz_format:
      format_name = "Ilyass pickle" if is_ilyass_pickle_format else "Ilyass npz"
      print(f"[Loader] Using {format_name} format loader.")
      # Use zero offsets for Ilyass pickle/npz format
      # ilyass_obj_pos_offset = np.array([0.0, 0.0, 0.0])
      # ilyass_obj_rot_offset = np.array([1.0, 0.0, 0.0, 0.0])

      # Extract data from pickle/npz format
      root_pos = np.asarray(data["root_pos"], dtype=np.float32)  # (T, 3)
      root_rot_wxyz = np.asarray(
        data["root_rot"], dtype=np.float32
      )  # (T, 4) in xyzw format
      root_rot_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]
      dof_pos = np.asarray(data["dof_pos"], dtype=np.float32)  # (T, 29)
      obj_pos = np.asarray(data["object_pos"], dtype=np.float32)  # (T, 3)
      obj_rot_wxyz = np.asarray(
        data["object_rot"], dtype=np.float32
      )  # (T, 4) in xyzw format
      obj_rot_xyzw = obj_rot_wxyz[:, [1, 2, 3, 0]]
      fps = (
        int(data["fps"][0]) if isinstance(data["fps"], np.ndarray) else int(data["fps"])
      )

      T = root_pos.shape[0]

      # Convert root rotation from xyzw to wxyz format
      root_rot_wxyz = root_rot_xyzw[:, [3, 0, 1, 2]]  # Convert xyzw -> wxyz
      # Normalize quaternions
      root_rot_wxyz = np.array([normalize_quat_wxyz(q) for q in root_rot_wxyz])

      # Convert root quaternion to axis-angle
      print("[Loader] Converting root quaternions to axis-angle...")
      root_quats_xyzw = root_rot_wxyz[:, [1, 2, 3, 0]]  # Convert wxyz -> xyzw for scipy
      root_rot_axis_angle = []
      for quat in root_quats_xyzw:
        quat_norm = quat / (np.linalg.norm(quat) + 1e-8)
        rotation = Rotation.from_quat(quat_norm)
        rot_vec = rotation.as_rotvec()
        root_rot_axis_angle.append(rot_vec)
      root_rot_axis_angle = np.array(root_rot_axis_angle)  # (T, 3)

      # Extract joints
      joint_dof = dof_pos  # (T, num_joints)
      num_joints = joint_dof.shape[1]

      # Ilyass pickle/npz format doesn't have PD targets, set to zero
      joint_pd_targets = np.zeros_like(joint_dof)
      print(f"[Loader] PD targets not found in {format_name} format, setting to zero.")

      # Convert object rotation from xyzw to wxyz format
      obj_rot_wxyz = obj_rot_xyzw[:, [3, 0, 1, 2]]  # Convert xyzw -> wxyz
      # Normalize quaternions
      obj_rot_wxyz = np.array([normalize_quat_wxyz(q) for q in obj_rot_wxyz])

      # Apply object position offset
      object_pos = obj_pos - OBJECT_POS_OFFSET

      # Apply quaternion offset (ilyass_obj_rot_offset is in wxyz format)
      obj_rot_offset_wxyz = normalize_quat_wxyz(OBJECT_ROT_OFFSET)
      # Multiply each quaternion with the offset: result = q * offset
      obj_rot_wxyz_offset = np.array(
        [
          normalize_quat_wxyz(quat_mul_wxyz(obj_rot_wxyz[i], obj_rot_offset_wxyz))
          for i in range(obj_rot_wxyz.shape[0])
        ]
      )
      print("[Loader] Applying object pos & quaternion offset...")

      # Convert object quaternion to axis-angle
      print("[Loader] Converting object quaternions to axis-angle...")
      obj_quats_xyzw = obj_rot_wxyz_offset[
        :, [1, 2, 3, 0]
      ]  # Convert wxyz -> xyzw for scipy
      object_rot_axis_angle = []
      for quat in obj_quats_xyzw:
        quat_norm = quat / (np.linalg.norm(quat) + 1e-8)
        rotation = Rotation.from_quat(quat_norm)
        rot_vec = rotation.as_rotvec()
        object_rot_axis_angle.append(rot_vec)
      object_rot_axis_angle = np.array(object_rot_axis_angle)  # (T, 3)

    else:
      raise ValueError(
        "Unsupported format. Expected SBTO, OmniRetarget, Ilyass pickle, or Ilyass npz format."
      )

    # Combine into frames format: [root_pos(3), axis_angle(3), joints(num_joints), joint_pd_targets(num_joints), obj_pos(3), obj_axis_angle(3)]
    frames = np.concatenate(
      [
        root_pos,
        root_rot_axis_angle,
        joint_dof,
        joint_pd_targets,
        object_pos,
        object_rot_axis_angle,
      ],
      axis=1,
    )

    return frames, num_joints, fps

  # Load and process motions
  if is_batch:
    print(f"\nProcessing {num_motions} motions...")
    all_processed_frames = []
    num_joints: int = 0  # Initialize for type checking
    fps: float = 0.0  # Initialize for type checking

    for motion_idx in range(num_motions):
      print(f"\n--- Processing motion {motion_idx + 1}/{num_motions} ---")
      frames, num_joints, fps = load_single_motion(data, motion_idx)

      original_duration = frames.shape[0] / fps
      print(f"Loaded motion with shape: {frames.shape}")
      print(f"FPS: {fps}")
      print(
        f"Original duration: {original_duration:.2f} seconds ({frames.shape[0]} frames)"
      )

      # Apply frame transformation: first frame at (x, y) = 0, yaw = 0
      print(
        "\nApplying frame transformation to set first frame at origin (x, y) = 0, yaw = 0..."
      )
      frames = transform_to_first_frame_origin(frames, num_joints)

      # Process the motion (duration, transitions, padding)
      processed_frames = process_single_motion(
        frames,
        num_joints,
        fps,
        duration,
        pad_duration,
        transition_duration,
        add_start_transition,
        add_end_transition,
      )

      # Append reversed trajectory if requested
      if add_reverse:
        print(
          f"\nAppending reversed trajectory for motion {motion_idx + 1}/{num_motions}..."
        )
        reversed_frames = reverse_motion_frames(processed_frames)
        processed_frames = np.vstack([processed_frames, reversed_frames])
        print(
          f"  Original: {processed_frames.shape[0] - reversed_frames.shape[0]} frames"
        )
        print(f"  Reversed: {reversed_frames.shape[0]} frames")
        print(f"  Total: {processed_frames.shape[0]} frames")

      all_processed_frames.append(processed_frames)

    # Stack all motions back into 3D array
    frames = np.stack(
      all_processed_frames, axis=0
    )  # (num_motions, num_frames, features)
    print(f"\nFinal batch shape: {frames.shape}")
  else:
    # Single motion case
    frames, num_joints, fps = load_single_motion(data, None)

    original_duration = frames.shape[0] / fps
    print(f"Loaded motion with shape: {frames.shape}")
    print(f"FPS: {fps}")
    print(
      f"Original duration: {original_duration:.2f} seconds ({frames.shape[0]} frames)"
    )

    # Apply frame transformation: first frame at (x, y) = 0, yaw = 0
    print(
      "\nApplying frame transformation to set first frame at origin (x, y) = 0, yaw = 0..."
    )
    frames = transform_to_first_frame_origin(frames, num_joints)
    print("Frame transformation applied.")

    # Process the motion (duration, transitions, padding)
    frames = process_single_motion(
      frames,
      num_joints,
      fps,
      duration,
      pad_duration,
      transition_duration,
      add_start_transition,
      add_end_transition,
    )

    # Append reversed trajectory if requested
    if add_reverse:
      print("\nAppending reversed trajectory...")
      original_frames = frames.shape[0]
      reversed_frames = reverse_motion_frames(frames)
      frames = np.vstack([frames, reversed_frames])
      print(f"  Original: {original_frames} frames")
      print(f"  Reversed: {reversed_frames.shape[0]} frames")
      print(f"  Total: {frames.shape[0]} frames")

    print(f"Final motion shape: {frames.shape}")

  # Convert axis-angle to quaternions and save
  if is_batch:
    # Process each motion in the batch
    print("\nConverting rotations from axis-angle to quaternions (XYZW)...")
    all_csv_data = []

    # Ensure num_joints is set (it should be from the processing loop above)
    assert num_joints > 0, "num_joints must be set before processing batch"

    for motion_idx in range(frames.shape[0]):
      motion_frames = frames[motion_idx]  # (num_frames, features)

      # Parse the data.
      root_pos = motion_frames[:, 0:3]  # (N, 3)
      root_rot_3d = motion_frames[:, 3:6]  # (N, 3) - axis-angle
      joint_dof = motion_frames[:, 6 : 6 + num_joints]  # (N, num_joints)
      pd_targets_csv = motion_frames[
        :, 6 + num_joints : 6 + 2 * num_joints
      ]  # (N, num_joints)
      # Extract object data
      obj_pos_start_idx = 6 + 2 * num_joints
      obj_pos_end_idx = obj_pos_start_idx + 3
      obj_rot_start_idx = obj_pos_end_idx
      obj_rot_end_idx = obj_rot_start_idx + 3
      obj_pos_csv = motion_frames[:, obj_pos_start_idx:obj_pos_end_idx]  # (N, 3)
      obj_rot_3d = motion_frames[
        :, obj_rot_start_idx:obj_rot_end_idx
      ]  # (N, 3) - axis-angle

      # Convert axis-angle rotation to quaternion (XYZW format).
      quats = []
      for rot_vec in root_rot_3d:
        angle = np.linalg.norm(rot_vec)
        if angle > 1e-6:
          rotation = Rotation.from_rotvec(rot_vec)
        else:
          rotation = Rotation.from_quat([0, 0, 0, 1])
        quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
        quats.append(quat_xyzw)
      quats = np.array(quats)  # (N, 4) in XYZW format.

      # Convert object rotations to quaternions
      obj_quats = []
      for rot_vec in obj_rot_3d:
        angle = np.linalg.norm(rot_vec)
        if angle > 1e-6:
          rotation = Rotation.from_rotvec(rot_vec)
        else:
          rotation = Rotation.from_quat([0, 0, 0, 1])
        quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
        obj_quats.append(quat_xyzw)
      obj_quats_csv = np.array(obj_quats)  # (N, 4) in XYZW format.

      # Combine: [root_pos(3), quat_xyzw(4), joint_dof(N), joint_pd_targets(N), obj_pos(3), obj_quat_xyzw(4)].
      csv_data = np.concatenate(
        [root_pos, quats, joint_dof, pd_targets_csv, obj_pos_csv, obj_quats_csv], axis=1
      )
      all_csv_data.append(csv_data)

    # Stack back to 3D: (num_motions, num_frames, features)
    csv_data = np.stack(all_csv_data, axis=0)
    print(f"Final batch CSV shape: {csv_data.shape}")

    # Save as NPZ to preserve 3D shape
    output_file = (
      csv_file if csv_file.endswith(".npz") else csv_file.replace(".csv", ".npz")
    )
    print(f"\nSaving to {output_file} (NPZ format to preserve 3D shape)...")
    np.savez(output_file, motion_data=csv_data)
    print("Done!")

    # Print some stats.
    print("\nMotion stats:")
    print(f"  Number of motions: {csv_data.shape[0]}")
    print(f"  Frames per motion: {csv_data.shape[1]}")
    print(f"  Features per frame: {csv_data.shape[2]}")
    print(f"  Duration per motion: {csv_data.shape[1] / fps:.2f} seconds")
    print(f"  FPS: {fps}")
    print(
      "\nOutput format: [x, y, z, qx, qy, qz, qw, joint1, joint2, ..., pd_target1, pd_target2, ..., obj_x, obj_y, obj_z, obj_qx, obj_qy, obj_qz, obj_qw]"
    )
  else:
    # Single motion case
    # Parse the data.
    root_pos = frames[:, 0:3]  # (N, 3)
    root_rot_3d = frames[:, 3:6]  # (N, 3) - axis-angle
    joint_dof = frames[:, 6 : 6 + num_joints]  # (N, num_joints)
    pd_targets_csv = frames[:, 6 + num_joints : 6 + 2 * num_joints]  # (N, num_joints)
    # Extract object data
    obj_pos_start_idx = 6 + 2 * num_joints
    obj_pos_end_idx = obj_pos_start_idx + 3
    obj_rot_start_idx = obj_pos_end_idx
    obj_rot_end_idx = obj_rot_start_idx + 3
    obj_pos_csv = frames[:, obj_pos_start_idx:obj_pos_end_idx]  # (N, 3)
    obj_rot_3d = frames[:, obj_rot_start_idx:obj_rot_end_idx]  # (N, 3) - axis-angle

    print(f"\nRoot position: {root_pos.shape}")
    print(f"Root rotation (axis-angle): {root_rot_3d.shape}")
    print(f"Joint DOF: {joint_dof.shape}")
    print(f"PD targets: {pd_targets_csv.shape}")
    print(f"Object position: {obj_pos_csv.shape}")
    print(f"Object rotation (axis-angle): {obj_rot_3d.shape}")

    # Convert axis-angle rotation to quaternion (XYZW format).
    print("\nConverting rotations from axis-angle to quaternions (XYZW)...")
    quats = []
    for rot_vec in root_rot_3d:
      angle = np.linalg.norm(rot_vec)
      if angle > 1e-6:
        rotation = Rotation.from_rotvec(rot_vec)
      else:
        rotation = Rotation.from_quat([0, 0, 0, 1])

      # Get quaternion in XYZW format.
      quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
      quats.append(quat_xyzw)

    quats = np.array(quats)  # (N, 4) in XYZW format.
    print(f"Root quaternions (XYZW): {quats.shape}")

    # Convert object rotations to quaternions
    print(
      "[Loader] Converting object rotations from axis-angle to quaternions (XYZW)..."
    )
    obj_quats = []
    for rot_vec in obj_rot_3d:
      angle = np.linalg.norm(rot_vec)
      if angle > 1e-6:
        rotation = Rotation.from_rotvec(rot_vec)
      else:
        rotation = Rotation.from_quat([0, 0, 0, 1])

      # Get quaternion in XYZW format.
      quat_xyzw = rotation.as_quat()  # Returns [x, y, z, w]
      obj_quats.append(quat_xyzw)

    obj_quats_csv = np.array(obj_quats)  # (N, 4) in XYZW format.
    print(f"Object quaternions (XYZW): {obj_quats_csv.shape}")

    # Combine: [root_pos(3), quat_xyzw(4), joint_dof(N), joint_pd_targets(N), obj_pos(3), obj_quat_xyzw(4)].
    csv_data = np.concatenate(
      [root_pos, quats, joint_dof, pd_targets_csv, obj_pos_csv, obj_quats_csv], axis=1
    )
    print(
      f"Final CSV shape: {csv_data.shape} (columns: 3 pos + 4 quat_xyzw + {num_joints} joints + {num_joints} joint_pd_targets + 3 obj_pos + 4 obj_quat_xyzw)"
    )

    # Save to CSV.
    print(f"\nSaving to {csv_file}...")
    np.savetxt(csv_file, csv_data, delimiter=",", fmt="%.8f")
    print("Done!")

    # Print some stats.
    print("\nMotion stats:")
    print(f"  Duration: {csv_data.shape[0] / fps:.2f} seconds")
    print(f"  Frames: {csv_data.shape[0]}")
    print(f"  FPS: {fps}")
    print(
      "\nOutput format: [x, y, z, qx, qy, qz, qw, joint1, joint2, ..., pd_target1, pd_target2, ..., obj_x, obj_y, obj_z, obj_qx, obj_qy, obj_qz, obj_qw]"
    )


if __name__ == "__main__":
  run_cli_function(main, description=__doc__)
