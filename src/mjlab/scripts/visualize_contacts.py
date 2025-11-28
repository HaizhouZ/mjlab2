"""Visualize robot trajectory, object trajectory, and contact information in MuJoCo.

This script loads a motion NPZ file with contact data and visualizes:
- Robot trajectory (animated)
- Object trajectory (animated)
- Contact locations (as spheres/markers)
- Contact boolean indicators (color-coded)
"""

import time
from typing import Any

import mujoco
import mujoco.viewer
import numpy as np
import torch
import tyro
import warp as wp

from mjlab.scene import Scene
from mjlab.sim.sim import Simulation, SimulationCfg
from mjlab.tasks.tracking.config.g1.env_cfgs import unitree_g1_flat_tracking_env_cfg_box

# Suppress Warp kernel loading messages for cleaner output
wp.config.quiet = True


def load_motion_data(npz_file: str) -> dict[str, Any]:
  """Load motion data from NPZ file."""
  data = np.load(npz_file, allow_pickle=True)

  # Extract required data
  result = {
    "fps": int(data["fps"][0]) if "fps" in data else 50,
    "joint_pos": data["joint_pos"],
    "joint_vel": data["joint_vel"],
    "body_pos_w": data["body_pos_w"],
    "body_quat_w": data["body_quat_w"],
    "body_lin_vel_w": data["body_lin_vel_w"],
    "body_ang_vel_w": data["body_ang_vel_w"],
  }
  # Optional object data
  if "object_pos_w" in data:
    result["object_pos_w"] = data["object_pos_w"]
    result["object_quat_w"] = data["object_quat_w"]
    result["object_lin_vel_w"] = data["object_lin_vel_w"]
    result["object_ang_vel_w"] = data["object_ang_vel_w"]

  # Optional contact data
  if "contact_positions" in data:
    result["contact_positions"] = data["contact_positions"]
    result["contact_indicators"] = data["contact_indicators"]

  return result


def set_robot_state(
  robot: Any,
  frame_data: dict[str, np.ndarray],
  device: str = "cpu",
  env_origin: np.ndarray | None = None,
) -> None:
  """Set robot state using entity write methods."""

  # Get root state (first body is root)
  root_pos = frame_data["body_pos_w"][0].copy()  # (3,)
  root_quat = frame_data["body_quat_w"][0].copy()  # (4,)

  # Get velocities if available, otherwise use zeros
  if "body_lin_vel_w" in frame_data:
    root_lin_vel = frame_data["body_lin_vel_w"][0].copy()  # (3,)
  else:
    root_lin_vel = np.zeros(3, dtype=np.float32)

  if "body_ang_vel_w" in frame_data:
    root_ang_vel = frame_data["body_ang_vel_w"][0].copy()  # (3,)
  else:
    root_ang_vel = np.zeros(3, dtype=np.float32)

  # Remove env_origin offset if present (positions in motion file may include it)
  if env_origin is not None:
    root_pos[:2] -= env_origin[:2]  # Only x, y are offset

  # Construct root state: [pos(3), quat(4), lin_vel(3), ang_vel(3)] = 13
  root_state = (
    torch.from_numpy(np.concatenate([root_pos, root_quat, root_lin_vel, root_ang_vel]))
    .float()
    .unsqueeze(0)
    .to(device)
  )  # (1, 13)

  robot.write_root_state_to_sim(root_state)

  # Set joint positions and velocities
  joint_pos = (
    torch.from_numpy(frame_data["joint_pos"]).float().unsqueeze(0).to(device)
  )  # (1, n_joints)

  if "joint_vel" in frame_data:
    joint_vel = (
      torch.from_numpy(frame_data["joint_vel"]).float().unsqueeze(0).to(device)
    )  # (1, n_joints)
  else:
    joint_vel = torch.zeros_like(joint_pos)

  robot.write_joint_state_to_sim(joint_pos, joint_vel)


def set_object_state(
  box: Any,
  object_pos: np.ndarray,
  object_quat: np.ndarray,
  device: str = "cpu",
  env_origin: np.ndarray | None = None,
) -> None:
  """Set object state using entity write methods."""

  # Remove env_origin offset if present
  obj_pos = object_pos.copy()
  if env_origin is not None:
    obj_pos[:2] -= env_origin[:2]  # Only x, y are offset

  # Construct root pose: [pos(3), quat(4)] = 7
  root_pose = (
    torch.from_numpy(np.concatenate([obj_pos, object_quat]))
    .float()
    .unsqueeze(0)
    .to(device)
  )  # (1, 7)

  box.write_root_link_pose_to_sim(root_pose)


def visualize_contacts(
  npz_file: str,
  device: str = "cpu",
  playback_speed: float = 1.0,
  show_contact_markers: bool = True,
  contact_marker_size: float = 0.05,
) -> None:
  """Visualize robot and object trajectories with contact information.

  Args:
    npz_file: Path to motion NPZ file with contact data
    device: Device to use (cpu or cuda)
    playback_speed: Playback speed multiplier (1.0 = normal speed)
    show_contact_markers: Whether to show contact location markers
    contact_marker_size: Size of contact markers in meters
  """
  print(f"Loading motion data from: {npz_file}")
  motion_data = load_motion_data(npz_file)

  num_frames = motion_data["body_pos_w"].shape[0]
  fps = motion_data["fps"]
  dt = 1.0 / fps

  print(f"Loaded {num_frames} frames at {fps} fps")
  print(f"Duration: {num_frames * dt:.2f} seconds")

  # Check for contact data
  has_contacts = (
    "contact_positions" in motion_data and "contact_indicators" in motion_data
  )
  if has_contacts:
    contact_positions = motion_data["contact_positions"]  # (T, num_eefs, 3)
    contact_indicators = motion_data["contact_indicators"]  # (T, num_eefs)
    num_eefs = contact_indicators.shape[1]
    total_contacts = contact_indicators.sum()
    print(
      f"Contact data found: {total_contacts} contact detections across {num_eefs} end effectors"
    )
  else:
    print("No contact data found in file")
    contact_positions = None
    contact_indicators = None

  # Check for object data
  has_object = "object_pos_w" in motion_data
  if has_object:
    print("Object trajectory found")
  else:
    print("No object trajectory found")

  # Setup simulation
  print("Setting up simulation...")
  sim_cfg = SimulationCfg()
  sim_cfg.mujoco.timestep = dt

  scene = Scene(unitree_g1_flat_tracking_env_cfg_box().scene, device=device)

  # End effector names
  eef_names = ["left_wrist_yaw_link", "right_wrist_yaw_link"]

  # Add contact visualization sites to the scene spec before compilation
  contact_site_ids = []
  num_eefs = 0
  if show_contact_markers and has_contacts and contact_indicators is not None:
    num_eefs = contact_indicators.shape[1]
    print(f"Adding {num_eefs} contact visualization sites...")
    for eef_idx in range(num_eefs):
      eef_name = eef_names[eef_idx] if eef_idx < len(eef_names) else f"eef_{eef_idx}"
      site_name = f"contact_{eef_name}"
      # Add site to worldbody for contact visualization
      # site = scene.spec.worldbody.add_site(
      #   name=site_name,
      #   pos=(0, 0, -10),  # Start hidden below ground
      #   size=(contact_marker_size,) * 3,
      #   type=mujoco.mjtGeom.mjGEOM_SPHERE,
      #   rgba=(1.0, 0.0, 0.0, 0.8)
      #   if eef_idx == 0
      #   else (0.0, 0.0, 1.0, 0.8),  # Red for left, blue for right
      #   group=3,  # Use group 3 for contact visualization
      # )
      contact_site_ids.append(site_name)

  model = scene.compile()
  sim = Simulation(num_envs=1, cfg=sim_cfg, model=model, device=device)
  scene.initialize(sim.mj_model, sim.model, sim.data)

  robot = scene["robot"]
  box = scene.entities.get("box") if hasattr(scene, "entities") else None

  mj_model = sim.mj_model
  mj_data = sim.mj_data

  # Get site IDs for contact markers
  contact_site_indices = []
  if show_contact_markers and has_contacts:
    for site_name in contact_site_ids:
      site_id = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_SITE, site_name)
      if site_id >= 0:
        contact_site_indices.append(site_id)
      else:
        contact_site_indices.append(-1)

  # Control state
  paused = {"active": False}
  step = {"current": 0}
  step_delta = {"value": 0}

  def key_callback(keycode: int):
    """Handle keyboard input."""
    # Space: toggle pause
    if keycode == 32:
      paused["active"] = not paused["active"]
    # Left arrow: step backward
    elif keycode == 263:
      step_delta["value"] = -1
      paused["active"] = True
    # Right arrow: step forward
    elif keycode == 262:
      step_delta["value"] = 1
      paused["active"] = True
    # 'R': reset to beginning
    elif keycode == 114:  # 'r'
      step["current"] = 0
      paused["active"] = True

  print("\nControls:")
  print("  Space: Pause/Resume")
  print("  Left Arrow: Step backward")
  print("  Right Arrow: Step forward")
  print("  R: Reset to beginning")
  print("\nStarting visualization...")

  with mujoco.viewer.launch_passive(
    mj_model, mj_data, key_callback=key_callback
  ) as viewer:
    # Add contact markers as sites (if supported) or use overlay
    contact_site_ids = []
    if show_contact_markers and has_contacts:
      # Try to add sites for contact visualization
      # Note: We'll use overlay rendering instead since adding sites dynamically is complex
      pass

    frame_idx = 0
    last_time = time.time()

    while viewer.is_running():
      current_time = time.time()

      # Handle step requests
      if step_delta["value"] != 0:
        frame_idx = (frame_idx + step_delta["value"]) % num_frames
        step_delta["value"] = 0
        step["current"] = frame_idx
      else:
        step["current"] = frame_idx

      # Update frame if not paused
      if not paused["active"]:
        elapsed = current_time - last_time
        if elapsed >= dt / playback_speed:
          frame_idx = (frame_idx + 1) % num_frames
          last_time = current_time
      else:
        time.sleep(0.01)  # Small sleep when paused
        continue

      # Get frame data
      frame_data = {
        "body_pos_w": motion_data["body_pos_w"][frame_idx],
        "body_quat_w": motion_data["body_quat_w"][frame_idx],
        "joint_pos": motion_data["joint_pos"][frame_idx],
      }

      # Add optional velocity data if available
      if "body_lin_vel_w" in motion_data:
        frame_data["body_lin_vel_w"] = motion_data["body_lin_vel_w"][frame_idx]
      if "body_ang_vel_w" in motion_data:
        frame_data["body_ang_vel_w"] = motion_data["body_ang_vel_w"][frame_idx]
      if "joint_vel" in motion_data:
        frame_data["joint_vel"] = motion_data["joint_vel"][frame_idx]

      # Get env origin (for offset correction)
      env_origin = (
        scene.env_origins[0].cpu().numpy() if hasattr(scene, "env_origins") else None
      )

      # Set robot state using entity methods
      set_robot_state(robot, frame_data, device=device, env_origin=env_origin)

      # Set object state if available
      if has_object and box is not None:
        object_pos = motion_data["object_pos_w"][frame_idx]
        object_quat = motion_data["object_quat_w"][frame_idx]
        set_object_state(
          box, object_pos, object_quat, device=device, env_origin=env_origin
        )

      # Forward kinematics (sync from warp to mujoco for visualization)
      sim.forward()

      # Copy state to mj_data for viewer
      mj_data.qpos[:] = sim.wp_data.qpos.numpy()[0]
      mj_data.qvel[:] = sim.wp_data.qvel.numpy()[0]
      mujoco.mj_forward(mj_model, mj_data)

      # Update contact visualization sites
      if (
        show_contact_markers
        and has_contacts
        and contact_indicators is not None
        and contact_positions is not None
      ):
        for eef_idx in range(num_eefs):
          if eef_idx < len(contact_site_indices) and contact_site_indices[eef_idx] >= 0:
            site_id = contact_site_indices[eef_idx]
            if contact_indicators[frame_idx, eef_idx]:
              contact_pos = contact_positions[frame_idx, eef_idx]
              if not np.isnan(contact_pos).any():
                # Update site position to show contact
                mj_data.site_xpos[site_id] = contact_pos
                # Make site visible (ensure alpha > 0)
                mj_model.site_rgba[site_id, 3] = 0.8  # Set alpha
              else:
                # Hide site if position is invalid
                mj_data.site_xpos[site_id, 2] = -10  # Move below ground
                mj_model.site_rgba[site_id, 3] = 0.0  # Make transparent
            else:
              # Hide site if no contact
              mj_data.site_xpos[site_id, 2] = -10  # Move below ground
              mj_model.site_rgba[site_id, 3] = 0.0  # Make transparent

        # Print contact info periodically (every 50 frames to avoid spam)
        if (
          frame_idx % 1 == 0
          and contact_indicators is not None
          and contact_positions is not None
        ):
          contact_info = []
          for eef_idx in range(num_eefs):
            if contact_indicators[frame_idx, eef_idx]:
              contact_pos = contact_positions[frame_idx, eef_idx]
              if not np.isnan(contact_pos).any():
                eef_name = (
                  eef_names[eef_idx] if eef_idx < len(eef_names) else f"EEF {eef_idx}"
                )
                contact_info.append(
                  f"{eef_name}: [{contact_pos[0]:.3f}, {contact_pos[1]:.3f}, {contact_pos[2]:.3f}]"
                )

          if contact_info:
            print(
              f"Frame {frame_idx}/{num_frames}: Contacts - {', '.join(contact_info)}"
            )

      # Sync viewer
      viewer.sync()

  print("Visualization closed.")


def main(
  npz_file: str,
  device: str = "cpu",
  playback_speed: float = 1.0,
  show_contact_markers: bool = True,
  contact_marker_size: float = 0.02,
) -> None:
  """Main entry point for contact visualization.

  Args:
    npz_file: Path to motion NPZ file with contact data
    device: Device to use (cpu or cuda)
    playback_speed: Playback speed multiplier (1.0 = normal speed)
    show_contact_markers: Whether to show contact location markers
    contact_marker_size: Size of contact markers in meters (default: 2cm)
  """
  visualize_contacts(
    npz_file=npz_file,
    device=device,
    playback_speed=playback_speed,
    show_contact_markers=show_contact_markers,
    contact_marker_size=contact_marker_size,
  )


if __name__ == "__main__":
  tyro.cli(main)
