"""Convert NPZ motion files (SBTO format) to CSV format."""

from typing import Any

import numpy as np
import tyro
from scipy.spatial.transform import Rotation, Slerp


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


def main(
  npz_file: str,
  csv_file: str,
  duration: float | None = None,
  pad_duration: float = 0.0,
  transition_duration: float = 1.0,
  add_start_transition: bool = False,
  add_end_transition: bool = False,
):
  """Convert NPZ motion file (Victor format) to CSV format.

  Output format: [root_pos(3), quat_xyzw(4), joint_dof(N), joint_pd_targets(N), obj_pos(3), obj_quat_xyzw(4)]

  Args:
    npz_file: Path to input .npz file (Victor format with base_xyz_quat and actuator_pos).
    csv_file: Path to output .csv file.
    duration: Desired duration in seconds. If None, use original duration. Motion will
      be cycled to reach this duration.
    pad_duration: Duration in seconds to hold the final pose at the end.
    transition_duration: Duration in seconds to blend to/from safe standing pose.
    add_start_transition: Whether to add transition from safe standing pose to motion start.
    add_end_transition: Whether to add transition from motion end to safe standing pose.
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

  OBJECT_POS_OFFSET = np.array(
    [0.0, 0.0, 0.0105]
  )  # from omniretarget largebox mesh (xyz)
  OBJECT_ROT_OFFSET = np.array(
    [0.00991298, 0.849052, -0.523456, 0.0707591]
  )  # from omniretarget largebox mesh (wxyz)

  print(f"Loading {npz_file}...")
  data = np.load(npz_file, allow_pickle=True)
  print("[Loader] Detected NPZ format.")

  # SBTO format
  is_sbto_format = "base_xyz_quat" in data and "actuator_pos" in data

  if not is_sbto_format:
    raise ValueError(
      f"Expected SBTO format NPZ file with 'base_xyz_quat' and 'actuator_pos' keys. "
      f"Found keys: {list(data.keys())}"
    )

  print("[Loader] Using SBTO format loader.")

  base_xyz_quat = data["base_xyz_quat"].astype(np.float32)
  actuator_pos = data["actuator_pos"].astype(np.float32)
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
  joint_pd_targets = data["u"].astype(np.float32)  # (N, num_joints)
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
  object_pos = obj[:, 0:3]  # (N, 3)
  object_pos = object_pos - OBJECT_POS_OFFSET

  # Extract object quaternions (in wxyz format from NPZ: [qw, qx, qy, qz])
  obj_quats_wxyz = obj[:, 3:7]  # (N, 4) - [qw, qx, qy, qz] = wxyz format

  # Apply quaternion offset (OBJECT_ROT_OFFSET is in wxyz format)
  print("[Loader] Applying object quaternion offset...")
  obj_rot_offset_wxyz = normalize_quat_wxyz(OBJECT_ROT_OFFSET)
  # Multiply each quaternion with the offset: result = q * offset (same as victor_npz_to_npz.py)
  obj_quats_wxyz_offset = np.array(
    [
      normalize_quat_wxyz(quat_mul_wxyz(obj_quats_wxyz[i], obj_rot_offset_wxyz))
      for i in range(obj_quats_wxyz.shape[0])
    ]
  )

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

  original_duration = frames.shape[0] / fps

  print(f"Loaded motion with shape: {frames.shape}")
  print(f"FPS: {fps}")
  print(
    f"Original duration: {original_duration:.2f} seconds ({frames.shape[0]} frames)"
  )

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

    print(f"New motion shape: {frames.shape}")
    print(
      f"New duration: {frames.shape[0] / fps:.2f} seconds ({frames.shape[0]} frames)"
    )

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
    print(f"After start transition, motion shape: {frames.shape}")
    print(f"Total duration so far: {frames.shape[0] / fps:.2f} seconds")

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
    print(f"After end transition, motion shape: {frames.shape}")
    print(f"Total duration: {frames.shape[0] / fps:.2f} seconds")

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
    print(f"Padded motion shape: {frames.shape}")
    print(
      f"Total duration: {frames.shape[0] / fps:.2f} seconds ({frames.shape[0]} frames)"
    )

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
  print("[Loader] Converting object rotations from axis-angle to quaternions (XYZW)...")
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
  tyro.cli(main)
