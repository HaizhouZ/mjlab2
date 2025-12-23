"""Unitree G1 flat tracking environment configurations."""

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs import ManagerBasedRlEnvCfg
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  MotionTrackingPDTargetsActionCfg,
)

from .object_env_cfgs import unitree_g1_flat_tracking_env_cfg_largebox


def unitree_g1_flat_tracking_env_cfg_largebox_pdtargets(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration with a box and PD targets."""
  cfg = unitree_g1_flat_tracking_env_cfg_largebox(
    has_state_estimation=has_state_estimation, play=play
  )

  ###
  # Motion Tracking PD Targets Action
  ###
  cfg.actions["joint_pos"] = MotionTrackingPDTargetsActionCfg(
    asset_name="robot",
    actuator_names=(".*",),
    scale=G1_ACTION_SCALE,
    command_name="motion",
  )

  cfg.rewards.pop("pd_tracking")

  return cfg


def unitree_g1_flat_tracking_env_cfg_largebox_default(
  has_state_estimation: bool = True,
  play: bool = False,
) -> ManagerBasedRlEnvCfg:
  """Create Unitree G1 flat terrain tracking configuration with a box and default action."""
  cfg = unitree_g1_flat_tracking_env_cfg_largebox(
    has_state_estimation=has_state_estimation, play=play
  )

  ###
  # Default Joint Position Action
  ###
  cfg.actions["joint_pos"] = JointPositionActionCfg(
    asset_name="robot",
    actuator_names=(".*",),
    scale=0.5,
    use_default_offset=True,
  )

  cfg.rewards.pop("pd_tracking")

  return cfg
