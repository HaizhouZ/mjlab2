"""Unitree G1 flat tracking environment configurations."""

import math

import mjlab.tasks.tracking.mdp as mdp
from mjlab.asset_zoo.robots import G1_ACTION_SCALE, get_box_cfg, get_g1_robot_cfg
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.managers.manager_term_config import (
  EventTermCfg,
  ObservationGroupCfg,
  ObservationTermCfg,
  RewardTermCfg,
  TerminationTermCfg,
)
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.sensor import ContactMatch, ContactSensorCfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg
from mjlab.tasks.tracking.tracking_env_cfg import make_tracking_env_cfg

_MAX_ANG_VEL = 500 * math.pi / 180.0  # [rad/s]


def unitree_g1_flat_tracking_env_cfg(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration."""
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

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)
  motion_cmd.anchor_body_name = "torso_link"
  motion_cmd.body_names = (
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
  )

  cfg.events["foot_friction"].params[
    "asset_cfg"
  ].geom_names = r"^(left|right)_foot[1-7]_collision$"
  cfg.events["base_com"].params["asset_cfg"].body_names = ("torso_link",)

  cfg.terminations["ee_body_pos"].params["body_names"] = (
    "left_ankle_roll_link",
    "right_ankle_roll_link",
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  cfg.viewer.body_name = "torso_link"

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
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_flat_tracking_env_cfg_box(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration with a box."""
  cfg = unitree_g1_flat_tracking_env_cfg(has_state_estimation=has_state_estimation)

  cfg.scene.entities = {"robot": get_g1_robot_cfg(), "box": get_box_cfg()}

  self_collision_cfg = ContactSensorCfg(
    name="self_collision",
    primary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    secondary=ContactMatch(mode="subtree", pattern="pelvis", entity="robot"),
    fields=("found",),
    reduce="none",
    num_slots=1,
  )

  left_eef_contact_sensor = ContactSensorCfg(
    name="left_eef_contact",
    primary=ContactMatch(mode="geom", pattern="left_wrist_collision", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="largebox_geom", entity="box"),
    fields=("found", "force", "pos"),
    reduce="netforce",
    num_slots=3,
  )

  right_eef_contact_sensor = ContactSensorCfg(
    name="right_eef_contact",
    primary=ContactMatch(mode="geom", pattern="right_wrist_collision", entity="robot"),
    secondary=ContactMatch(mode="geom", pattern="largebox_geom", entity="box"),
    fields=("found", "force", "pos"),
    reduce="netforce",
    num_slots=3,
  )

  cfg.scene.sensors = (
    self_collision_cfg,
    left_eef_contact_sensor,
    right_eef_contact_sensor,
  )

  ###
  # Add Motion Offset Joint Position Action
  ###
  joint_pos_action = cfg.actions["joint_pos"]
  assert isinstance(joint_pos_action, JointPositionActionCfg)
  joint_pos_action.scale = G1_ACTION_SCALE

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  motion_cmd.eef_body_names = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
  )

  ###
  # Object Tracking Reward Terms
  ###
  cfg.rewards["contact_match"] = RewardTermCfg(
    func=mdp.eef_contact_indicator_match,
    weight=1.25,
    params={
      "command_name": "motion",
      "eef_body_names": motion_cmd.eef_body_names,
      "sensor_names": ["left_eef_contact", "right_eef_contact"],
      "gain": 1.0,
      "force_threshold": 10.0,
      "force_penalty_std": 10.0,
    },
  )
  cfg.rewards["object_global_pos"] = RewardTermCfg(
    func=mdp.object_global_position_error_exp,
    weight=1.2,
    params={
      "command_name": "motion",
      "object_asset_cfg": SceneEntityCfg("box"),
      "std": 0.25,
    },
  )
  cfg.rewards["object_global_ori"] = RewardTermCfg(
    func=mdp.object_global_orientation_error_exp,
    weight=0.5,
    params={
      "command_name": "motion",
      "object_asset_cfg": SceneEntityCfg("box"),
      "std": 0.4,
    },
  )
  cfg.rewards["bad_termination"] = RewardTermCfg(
    func=mdp.is_terminated,
    weight=-100.0,
  )

  ###
  # Object Tracking Termination Terms
  ###
  cfg.terminations["base_ang_vel_exceed"] = TerminationTermCfg(
    func=mdp.base_ang_vel_exceed,
    params={"threshold": _MAX_ANG_VEL},
  )

  ###
  # Object Tracking Event Terms
  ###
  cfg.events["push_object"] = EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
      "asset_cfg": SceneEntityCfg("box"),
      "velocity_range": {
        "x": (-0.5, 0.5),
        "y": (-0.5, 0.5),
        "z": (-0.9, 0.9),
        "roll": (-1.5, 1.5),
        "pitch": (-1.5, 1.5),
        "yaw": (-1.5, 1.5),
      },
    },
  )

  cfg.events["hand_friction"] = EventTermCfg(
    func=mdp.randomize_field,
    mode="startup",
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", geom_names=("left_wrist_collision", "right_wrist_collision")
      ),
      "operation": "abs",
      "field": "geom_friction",
      "ranges": (0.3, 1.2),
    },
  )

  ###
  # Actor (Policy) Object Tracking Observation Terms
  ###
  cfg.observations["policy"].terms["object_global_pos"] = ObservationTermCfg(
    func=mdp.object_pos_b, params={"command_name": "motion"}
  )
  cfg.observations["policy"].terms["object_global_ori"] = ObservationTermCfg(
    func=mdp.object_orientation_error,
    params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box")},
  )

  ###
  # Critic Object Tracking Observation Terms
  ###
  cfg.observations["critic"].terms["object_global_pos"] = ObservationTermCfg(
    func=mdp.object_pos_b, params={"command_name": "motion"}
  )
  cfg.observations["critic"].terms["object_global_ori"] = ObservationTermCfg(
    func=mdp.object_orientation_error,
    params={"command_name": "motion", "asset_cfg": SceneEntityCfg("box")},
  )
  cfg.observations["critic"].terms["object_lin_vel_w"] = ObservationTermCfg(
    func=mdp.object_lin_vel_w, params={"asset_cfg": SceneEntityCfg("box")}
  )
  cfg.observations["critic"].terms["object_ang_vel_w"] = ObservationTermCfg(
    func=mdp.object_ang_vel_w, params={"asset_cfg": SceneEntityCfg("box")}
  )

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
    # Effectively infinite episode length.
    cfg.episode_length_s = int(1e9)

    cfg.observations["policy"].enable_corruption = False
    cfg.events.pop("push_robot", None)

    # Disable RSI randomization.
    motion_cmd.pose_range = {}
    motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg
