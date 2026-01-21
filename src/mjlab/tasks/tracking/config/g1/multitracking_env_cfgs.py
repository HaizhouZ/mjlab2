"""Unitree G1 flat tracking environment configurations."""

import os

import mjlab.tasks.tracking.mdp as mdp
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.managers.observation_manager import ObservationTermCfg

from .env_cfgs import unitree_g1_flat_tracking_env_cfg


def unitree_g1_flat_multitracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain multi-tracking configuration with multiple motion trajectories."""
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  assert cfg.commands is not None
  cfg.commands["motion"] = mdp.MultiMotionCommandCfg(
    entity_name="robot",
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
    eef_body_names=(
      "left_wrist_yaw_link",
      "right_wrist_yaw_link",
      "left_ankle_roll_link",
      "right_ankle_roll_link",
    ),
    motion_name_pattern=[".*"],
    debug_vis=True,
    wandb_entity="mim-atari",
    wandb_project="g1-multitracking",
    resampling_time_range=(1e9, 1e9),
    horizon=int(
      os.environ.get("MJLAB_MOTION_HORIZON", 1)
    ),  # Set horizon from environment variable or default to 1
    play=play,  # Pass play flag to command config
  )

  ###
  # Trajectory Encoding Observation Terms
  ###
  cfg.observations["policy"].terms["future_traj"] = ObservationTermCfg(
    func=mdp.trajectory_encoding,
    params={"command_name": "motion"},
  )

  cfg.observations["critic"].terms["future_traj"] = ObservationTermCfg(
    func=mdp.trajectory_encoding,
    params={"command_name": "motion"},
  )

  # Apply play mode overrides.
  if play:
    cfg.commands["motion"].sampling_mode = "start"
    cfg.commands["motion"].start_motion_name = "fight1_subject5"

  return cfg
