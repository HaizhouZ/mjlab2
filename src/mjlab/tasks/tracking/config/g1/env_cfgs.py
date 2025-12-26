"""Unitree G1 flat tracking environment configurations."""

import math

import mjlab.tasks.tracking.mdp as mdp
from mjlab.asset_zoo.robots import (
  G1_ACTION_SCALE,
  get_box_cfg,
  get_g1_robot_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  MotionTrackingJointPositionActionCfg,
)
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
from mjlab.utils.noise import UniformNoiseCfg as Unoise

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
    # cfg.events.pop("push_robot", None)
    # cfg.events.pop("base_com", None)
    # cfg.events.pop("add_joint_default_pos", None)
    # cfg.events.pop("foot_friction", None)
    # cfg.commands["motion"].joint_position_range = (-0.0, 0.0)

    # cfg.terminations["base_ang_vel_exceed"] = None
    # cfg.terminations["ee_body_pos"] = None
    # cfg.terminations["anchor_pos"] = None
    # cfg.terminations["anchor_ori"] = None

    # # Disable RSI randomization.
    # motion_cmd.pose_range = {}
    # motion_cmd.velocity_range = {}

    motion_cmd.sampling_mode = "start"

  return cfg


def unitree_g1_flat_tracking_env_cfg_box(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration with a box."""
  cfg = unitree_g1_flat_tracking_env_cfg(
    has_state_estimation=has_state_estimation, play=play
  )

  cfg.scene.entities = {"robot": get_g1_robot_cfg(), "object": get_box_cfg()}

  ###
  # Contact Sensors
  ###
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
    primary=ContactMatch(
      mode="geom", pattern="left_(elbow_yaw|wrist)_collision", entity="robot"
    ),
    secondary=ContactMatch(mode="geom", pattern="largebox_geom", entity="object"),
    fields=("found", "force", "pos"),
    reduce="none",
    num_slots=1,
  )

  right_eef_contact_sensor = ContactSensorCfg(
    name="right_eef_contact",
    primary=ContactMatch(
      mode="geom", pattern="right_(elbow_yaw|wrist)_collision", entity="robot"
    ),
    secondary=ContactMatch(mode="geom", pattern="largebox_geom", entity="object"),
    fields=("found", "force", "pos"),
    reduce="none",
    num_slots=1,
  )

  left_foot_contact_sensor = ContactSensorCfg(
    name="left_foot_contact",
    primary=ContactMatch(
      mode="geom", pattern="left_foot[1-7]_collision", entity="robot"
    ),
    secondary=ContactMatch(mode="geom", pattern="largebox_geom", entity="object"),
    fields=("found", "force", "pos"),
    reduce="none",
    num_slots=1,
  )

  right_foot_contact_sensor = ContactSensorCfg(
    name="right_foot_contact",
    primary=ContactMatch(
      mode="geom", pattern="right_foot[1-7]_collision", entity="robot"
    ),
    secondary=ContactMatch(mode="geom", pattern="largebox_geom", entity="object"),
    fields=("found", "force", "pos"),
    reduce="none",
    num_slots=1,
  )

  cfg.scene.sensors = (
    self_collision_cfg,
    left_eef_contact_sensor,
    right_eef_contact_sensor,
    left_foot_contact_sensor,
    right_foot_contact_sensor,
  )

  assert cfg.commands is not None
  motion_cmd = cfg.commands["motion"]
  assert isinstance(motion_cmd, MotionCommandCfg)

  motion_cmd.eef_body_names = (
    "left_wrist_yaw_link",
    "right_wrist_yaw_link",
    "left_ankle_roll_link",
    "right_ankle_roll_link",
  )
  # Object pose and velocity range for RSI
  motion_cmd.object_pose_range = {
    "x": (0.0, 0.2),
    "y": (-0.1, 0.1),
    "z": (0.0, 0.0),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-0.3, 0.3),
  }
  motion_cmd.object_velocity_range = {
    "x": (-0.5, 0.5),
    "y": (-0.5, 0.5),
    "z": (-0.5, 0.5),
    "roll": (-0.5, 0.5),
    "pitch": (-0.5, 0.5),
    "yaw": (-0.5, 0.5),
  }

  ###
  # Motion Tracking Joint Position Action
  ###
  cfg.actions["joint_pos"] = MotionTrackingJointPositionActionCfg(
    asset_name="robot",
    actuator_names=(".*",),
    scale=G1_ACTION_SCALE,
    command_name="motion",
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
      "sensor_names": [
        "left_eef_contact",
        "right_eef_contact",
        "left_foot_contact",
        "right_foot_contact",
      ],
      "force_threshold": 40.0,
      "force_penalty_std": 10.0,
    },
  )
  # cfg.rewards["object_relative_pos"] = RewardTermCfg(
  #   func=mdp.object_relative_position_error_exp,
  #   weight=1.0,
  #   params={
  #     "command_name": "motion",
  #     "object_asset_cfg": SceneEntityCfg("object"),
  #     "std": 0.2,
  #   },
  # )
  # cfg.rewards["object_relative_ori"] = RewardTermCfg(
  #   func=mdp.object_relative_orientation_error_exp,
  #   weight=1.2,
  #   params={
  #     "command_name": "motion",
  #     "object_asset_cfg": SceneEntityCfg("object"),
  #     "std": 0.3,
  #   },
  # )
  cfg.rewards["object_global_pos"] = RewardTermCfg(
    func=mdp.object_global_position_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "object_asset_cfg": SceneEntityCfg("object"),
      "std": 0.25,
    },
  )
  cfg.rewards["object_global_ori"] = RewardTermCfg(
    func=mdp.object_global_orientation_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "object_asset_cfg": SceneEntityCfg("object"),
      "std": 0.3,
    },
  )
  cfg.rewards["object_global_lin_vel"] = RewardTermCfg(
    func=mdp.object_global_linear_velocity_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "object_asset_cfg": SceneEntityCfg("object"),
      "std": 0.5,
    },
  )
  cfg.rewards["object_global_ang_vel"] = RewardTermCfg(
    func=mdp.object_global_angular_velocity_error_exp,
    weight=1.0,
    params={
      "command_name": "motion",
      "object_asset_cfg": SceneEntityCfg("object"),
      "std": 1.57,
    },
  )
  cfg.rewards["pd_tracking"] = RewardTermCfg(
    func=mdp.pd_tracking_error_exp,
    weight=2.0,
    params={
      "command_name": "motion",
      "std": 0.3,
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
  # # # Add termination for object pose deviation exceeding thresholds from reference
  # cfg.terminations["bad_object_pose"] = TerminationTermCfg(
  #   func=mdp.bad_object_pose,
  #   params={
  #     "command_name": "motion",
  #     "asset_cfg": SceneEntityCfg("object"),
  #     "threshold": {
  #       "roll": math.pi / 2.5,  # 72 degrees
  #       "pitch": math.pi / 2.5,  # 72 degrees
  #       "yaw": None,
  #       "pos_x": 1.0,
  #       "pos_y": 1.0,
  #       "pos_z": 0.3,
  #     },
  #   },
  # )
  # cfg.terminations["object_pos_z"] = TerminationTermCfg(
  #   func=mdp.object_pos_z,
  #   params={
  #     "command_name": "motion",
  #     "asset_cfg": SceneEntityCfg("object"),
  #     "threshold": 0.5,
  #   },
  # )

  # cfg.terminations["object_pos_exceed"] = TerminationTermCfg(
  #   func=mdp.object_too_far,
  #   params={
  #     "command_name": "motion",
  #     "asset_cfg": SceneEntityCfg("object"),
  #     "threshold": 0.5,
  #   },
  # )

  #   cfg.terminations["contact_mismatch"] = TerminationTermCfg(
  #   func=mdp.contact_mismatch_consecutive,
  #   params={
  #     "command_name": "motion",
  #     "eef_body_names": ["left_wrist_yaw_link", "right_wrist_yaw_link"],
  #     "sensor_names": ["left_eef_contact", "right_eef_contact"],
  #     "consecutive_steps": 15,
  #   },
  # )

  ###
  # Object Tracking Event Terms
  ###
  cfg.events["push_object"] = EventTermCfg(
    func=mdp.push_by_setting_velocity,
    mode="interval",
    interval_range_s=(1.0, 3.0),
    params={
      "asset_cfg": SceneEntityCfg("object"),
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

  # cfg.events["object_mass"] = EventTermCfg(
  #   func=mdp.randomize_field,
  #   mode="startup",
  #   params={
  #     "asset_cfg": SceneEntityCfg("object"),
  #     "operation": "scale",
  #     "field": "body_mass",
  #     "ranges": (0.6, 1.5),
  #   },
  # )

  cfg.events["hand_friction"] = EventTermCfg(
    func=mdp.randomize_field,
    mode="startup",
    # domain_randomization=True,
    params={
      "asset_cfg": SceneEntityCfg(
        "robot", geom_names=("left_wrist_collision", "right_wrist_collision")
      ),
      "operation": "abs",
      "field": "geom_friction",
      "ranges": (0.3, 1.2),
    },
  )

  # cfg.events["box_friction"] = EventTermCfg(
  #   func=mdp.randomize_field,
  #   domain_randomization=True,
  #   mode="startup",
  #   params={
  #     "asset_cfg": SceneEntityCfg("object", geom_names=("largebox_geom",)),
  #     "operation": "abs",
  #     "field": "geom_friction",
  #     "ranges": (0.2, 0.8),
  #   },
  # )

  # cfg.events["box_size"] = EventTermCfg(
  #   func=mdp.randomize_field,
  #   mode="startup",
  #   domain_randomization=True,
  #   params={
  #     "asset_cfg": SceneEntityCfg("object"),
  #     "operation": "scale",
  #     "field": "geom_size",
  #     "ranges": {
  #       0: (0.7, 1.3),
  #       1: (0.7, 1.3),
  #       2: (1.0, 1.0),
  #     },  # Only randomize axis 1 (breadth/y), keep length (0) and height (2) fixed
  #   },
  # )

  ###
  # Actor (Policy) Object Tracking Observation Terms
  ###
  cfg.observations["policy"].terms["object_pos_b"] = ObservationTermCfg(
    func=mdp.object_pos_b,
    noise=Unoise(n_min=-0.1, n_max=0.1),
    params={"command_name": "motion"},
  )
  # cfg.observations["policy"].terms["object_ori_b"] = ObservationTermCfg(
  #   func=mdp.object_ori_b,
  #   noise=Unoise(n_min=-0.05, n_max=0.05),
  #   params={"command_name": "motion"},
  # )
  # cfg.observations["policy"].terms["object_lin_vel_w"] = ObservationTermCfg(
  #   func=mdp.object_lin_vel_w,
  #   noise=Unoise(n_min=-0.25, n_max=0.25),
  #   params={"asset_cfg": SceneEntityCfg("object")},
  # )
  # cfg.observations["policy"].terms["object_ang_vel_w"] = ObservationTermCfg(
  #   func=mdp.object_ang_vel_w,
  #   noise=Unoise(n_min=-0.25, n_max=0.25),
  #   params={"asset_cfg": SceneEntityCfg("object")},
  # )
  cfg.observations["policy"].terms["object_pos_error"] = ObservationTermCfg(
    func=mdp.object_position_error,
    noise=Unoise(n_min=-0.05, n_max=0.05),
    params={"command_name": "motion", "asset_cfg": SceneEntityCfg("object")},
  )
  cfg.observations["policy"].terms["object_ori_error"] = ObservationTermCfg(
    func=mdp.object_orientation_error,
    noise=Unoise(n_min=-0.05, n_max=0.05),
    params={"command_name": "motion", "asset_cfg": SceneEntityCfg("object")},
  )

  ###
  # Critic Object Tracking Observation Terms
  ###
  cfg.observations["critic"].terms["object_pos_b"] = ObservationTermCfg(
    func=mdp.object_pos_b, history_length=10, params={"command_name": "motion"}
  )
  cfg.observations["critic"].terms["object_ori_b"] = ObservationTermCfg(
    func=mdp.object_ori_b, history_length=10, params={"command_name": "motion"}
  )
  cfg.observations["critic"].terms["object_lin_vel_w"] = ObservationTermCfg(
    func=mdp.object_lin_vel_w,
    history_length=10,
    params={"asset_cfg": SceneEntityCfg("object")},
  )
  cfg.observations["critic"].terms["object_ang_vel_w"] = ObservationTermCfg(
    func=mdp.object_ang_vel_w,
    history_length=10,
    params={"asset_cfg": SceneEntityCfg("object")},
  )
  cfg.observations["critic"].terms["object_pos_error"] = ObservationTermCfg(
    func=mdp.object_position_error,
    history_length=10,
    params={"command_name": "motion", "asset_cfg": SceneEntityCfg("object")},
  )
  cfg.observations["critic"].terms["object_ori_error"] = ObservationTermCfg(
    func=mdp.object_orientation_error,
    history_length=10,
    params={"command_name": "motion", "asset_cfg": SceneEntityCfg("object")},
  )
  # cfg.observations["critic"].terms["object_contact"] = ObservationTermCfg(
  #   func=mdp.contact_indicator,
  #   params={"command_name": "motion"},
  # )

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
    # cfg.events.pop("push_robot", None)
    # cfg.events.pop("push_object", None)
    # motion_cmd.pose_range = {}
    # motion_cmd.velocity_range = {}
    # motion_cmd.object_pose_range = {}
    # motion_cmd.object_velocity_range = {}

    # cfg.events.pop("push_object", None)

  return cfg
