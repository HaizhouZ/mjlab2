"""Unitree G1 flat tracking environment configurations."""

import mjlab.tasks.tracking.mdp as mdp
from mjlab.asset_zoo.robots import (
  get_g1_robot_cfg,
  get_largebox_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.manager_term_config import (
  ObservationTermCfg,
)
from mjlab.tasks.tracking.mdp import MultiMotionCommandCfg

from .env_cfgs import unitree_g1_flat_tracking_env_cfg_box


def unitree_g1_flat_multitracking_env_cfg_box(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration with a box."""
  cfg = unitree_g1_flat_tracking_env_cfg_box(
    has_state_estimation=has_state_estimation, play=play
  )

  assert cfg.commands is not None
  cfg.commands["motion"] = MultiMotionCommandCfg(
    asset_name="robot",
    anchor_body_name="torso_link",
    body_names=(
      "pelvis",
      "left_hip_roll_link",
      "left_knee_link",
      "left_ankle_roll_link",
      "right_hip_roll_link",
      "right_knee_link",
      "right_ankle_roll_link",
      "torso_link",
      "left_shoulder_roll_link",
      "left_elbow_link",
      "left_wrist_yaw_link",
      "right_shoulder_roll_link",
      "right_elbow_link",
      "right_wrist_yaw_link",
    ),
    pose_range={
      "x": (-0.05, 0.05),
      "y": (-0.05, 0.05),
      "z": (-0.01, 0.01),
      "roll": (-0.1, 0.1),
      "pitch": (-0.1, 0.1),
      "yaw": (-0.2, 0.2),
    },
    velocity_range={
      "x": (-0.5, 0.5),
      "y": (-0.5, 0.5),
      "z": (-0.2, 0.2),
      "roll": (-0.52, 0.52),
      "pitch": (-0.52, 0.52),
      "yaw": (-0.78, 0.78),
    },
    joint_position_range=(-0.1, 0.1),
    eef_body_names=(
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
      "left_ankle_roll_link",
      "right_ankle_roll_link",
    ),
    # motion_dir=str(motions_dir),
    encoder_dir="logs/trajectory_autoencoder/unet_simple/2025-12-17/14-04-49/best_model.jit",
    wandb_entity="ATARITUM",
    wandb_project="sbto_v1",
    motion_name_pattern=[".*"],
    debug_vis=True,
    resampling_time_range=(1e9, 1e9),
    horizon=32,
  )

  # Apply play mode overrides.
  if play:
    cfg.commands["motion"].motion_assignment_mode = "best"
    cfg.commands["motion"].sampling_mode = "start"

  return cfg


def unitree_g1_flat_multitracking_env_cfg_largebox(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain multi-tracking configuration with a large box."""
  cfg = unitree_g1_flat_multitracking_env_cfg_box(
    has_state_estimation=has_state_estimation, play=play
  )
  cfg.scene.entities = {"robot": get_g1_robot_cfg(), "object": get_largebox_cfg()}

  return cfg


def unitree_g1_flat_multitracking_encoding_env_cfg_box(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain multi-tracking configuration with a box and encoding."""
  cfg = unitree_g1_flat_multitracking_env_cfg_box(
    has_state_estimation=has_state_estimation, play=play
  )

  ###
  # Trajectory Encoding Observation Terms
  ###
  cfg.observations["policy"].terms["trajectory_encoding"] = ObservationTermCfg(
    func=mdp.trajectory_encoding,
    params={"command_name": "motion"},
  )

  cfg.observations["critic"].terms["trajectory_encoding"] = ObservationTermCfg(
    func=mdp.trajectory_encoding,
    params={"command_name": "motion"},
  )

  return cfg


def unitree_g1_flat_multitracking_encoding_env_cfg_largebox(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain multi-tracking configuration with a large box and encoding."""
  cfg = unitree_g1_flat_multitracking_encoding_env_cfg_box(
    has_state_estimation=has_state_estimation, play=play
  )
  cfg.scene.entities = {"robot": get_g1_robot_cfg(), "object": get_largebox_cfg()}

  return cfg
