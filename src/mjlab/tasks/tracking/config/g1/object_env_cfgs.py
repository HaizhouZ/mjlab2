"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import (
  get_cylinder_cfg,
  get_g1_robot_cfg,
  get_largebox_cfg,
  get_platform_cfg,
)
from mjlab.envs import ManagerBasedRlEnvCfg, mdp
from mjlab.managers.manager_term_config import EventTermCfg
from mjlab.managers.scene_entity_config import SceneEntityCfg

from .env_cfgs import unitree_g1_flat_tracking_env_cfg_box


def unitree_g1_flat_tracking_env_cfg_largebox(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration with a box and no state estimation."""
  cfg = unitree_g1_flat_tracking_env_cfg_box(
    has_state_estimation=has_state_estimation, play=play
  )
  cfg.scene.entities = {"robot": get_g1_robot_cfg(), "object": get_largebox_cfg()}

  # cfg.commands["motion"].object_pose_range = {
  #   "x": (0.0, 0.0),
  #   "y": (0.0, 0.0),
  #   "z": (0.2, 0.2),
  #   "roll": (0.0, 0.0),
  #   "pitch": (0.0, 0.0),
  #   "yaw": (0.0, 0.0),
  # }

  return cfg


def unitree_g1_flat_tracking_env_cfg_largebox_platform(
  has_state_estimation: bool = True, play: bool = False
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration with a box and no state estimation."""
  cfg = unitree_g1_flat_tracking_env_cfg_largebox(
    has_state_estimation=has_state_estimation, play=play
  )
  cfg.scene.entities = {
    "robot": get_g1_robot_cfg(),
    "platform": get_platform_cfg(),
    "object": get_largebox_cfg(),
  }
  cfg.events["platform_pos"] = EventTermCfg(
    func=mdp.reset_root_state_uniform,
    mode="reset",
    params={
      "asset_cfg": SceneEntityCfg("platform"),
      "pose_range": {"x": (0.0, 0.0), "y": (0.6, 0.75), "z": (0.6, 0.7)},
    },
  )

  return cfg


def unitree_g1_flat_tracking_env_cfg_cylinder(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration with a cylinder and no state estimation."""
  cfg = unitree_g1_flat_tracking_env_cfg_box(
    has_state_estimation=has_state_estimation, play=play
  )
  cfg.scene.entities = {"robot": get_g1_robot_cfg(), "object": get_cylinder_cfg()}

  # Only modify push_object event if it exists (it's removed in play mode)
  if "push_object" in cfg.events:
    cfg.events["push_object"].params["velocity_range"] = {
      "x": (-0.25, 0.25),
      "y": (-0.25, 0.25),
      "z": (-0.5, 0.5),
      "roll": (-0.5, 0.5),
      "pitch": (-0.5, 0.5),
      "yaw": (-0.5, 0.5),
    }

  cfg.rewards.pop("pd_tracking")

  return cfg
