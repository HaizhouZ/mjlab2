"""Unitree G1 flat tracking environment configurations with YAML support."""

from dataclasses import dataclass, field
from typing import Optional

from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.observation_manager import ObservationGroupCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg
from mjlab.utils import override_dataclass_from_source


def unitree_g1_flat_tracking_env_cfg(
  anchor_body_name: str = "torso_link",
  body_names: tuple = (
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
  has_state_estimation: bool = True,
  play: bool = False,
  config_path: Optional[str] = None,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration.

  Args:
    motion_file: Path to motion capture data file (.npz).
    anchor_body_name: Reference body for motion tracking (default: "torso_link").
    body_names: G1 skeleton bodies to track (14 bodies by default).
    eef_body_names: End effector bodies for termination (4 by default).
    entity_name: Robot entity name (default: "robot").
    has_state_estimation: Include state estimation in observations (default: True).
    play: Play mode - infinite episodes, no randomization (default: False).
    config_path: Optional YAML/JSON config file path. Only fields in the file
      override defaults; missing fields use parameter defaults.

  Returns:
    ManagerBasedRlEnvCfg: Complete environment configuration.

  Example:
    cfg = unitree_g1_flat_tracking_env_cfg(config_path="config.yaml")
  """
  # Create base environment config
  cfg = make_tracking_env_cfg()

  cfg.scene.entities = {"robot": get_g1_robot_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )
  cfg.scene.sensors = (self_collision_cfg,)

  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  # Create default motion command config from function parameters
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  motion_cmd.anchor_body_name = anchor_body_name
  motion_cmd.body_names = body_names

  # Override from config file if provided
  if config_path:
    override_dataclass_from_source(motion_cmd, config_path)

  cfg.commands["motion"] = motion_cmd

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = motion_cmd.anchor_body_name

  # Modify observations if we don't have state estimation.
  if not has_state_estimation:
    new_policy_terms = {
      k: v
      for k, v in cfg.observations["policy"].terms.items()
      if k not in ["motion_anchor_pos_b", "base_lin_vel"]
    }
    cfg.observations["policy"] = ObservationGroupCfg(
      terms=new_policy_terms,
      concatenate_terms=True,
      enable_corruption=True,
    )

  # Apply play mode overrides.
  if play:
    cfg.is_play = True
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)
    # cfg.events.pop("base_com", None)
    # cfg.events.pop("add_joint_default_pos", None)
    # cfg.events.pop("foot_friction", None)
    # cfg.events["push_robot"].params["visualize"] = True

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
