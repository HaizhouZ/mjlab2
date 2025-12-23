"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  get_cylinder_cfg,
  get_g1_robot_cfg,
  get_largebox_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg

from .env_cfgs import unitree_g1_flat_tracking_env_cfg_box


def unitree_g1_flat_tracking_env_cfg_largebox(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration with a box and no state estimation."""
  cfg = unitree_g1_flat_tracking_env_cfg_box(
    has_state_estimation=has_state_estimation, play=play
  )
  cfg.scene.entities = {"robot": get_g1_robot_cfg(), "box": get_largebox_cfg()}

  # cfg.commands["motion"].object_pose_range = {
  #   "x": (0.0, 0.0),
  #   "y": (0.0, 0.0),
  #   "z": (0.2, 0.2),
  #   "roll": (0.0, 0.0),
  #   "pitch": (0.0, 0.0),
  #   "yaw": (0.0, 0.0),
  # }

  return cfg


def unitree_g1_flat_tracking_env_cfg_cylinder(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration with a cylinder and no state estimation."""
  cfg = unitree_g1_flat_tracking_env_cfg_box(
    has_state_estimation=has_state_estimation, play=play
  )
  cfg.scene.entities = {"robot": get_g1_robot_cfg(), "box": get_cylinder_cfg()}

  cfg.rewards.pop("pd_tracking")
  cfg.commands["motion"].object_pose_range = {
    "x": (-0.1, 0.1),
    "y": (0.0, 0.0),
    "z": (0.0, 0.1),
    "roll": (0.0, 0.0),
    "pitch": (0.0, 0.0),
    "yaw": (-0.17, 0.17),
  }

  return cfg
