from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.action_manager import ActionTerm
from mjlab.utils.lab_api.string import (
  resolve_matching_names_values,
)

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv
  from mjlab.envs.mdp.actions import actions_config


class JointAction(ActionTerm):
  """Base class for joint actions."""

  _asset: Entity

  def __init__(self, cfg: actions_config.JointActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

    joint_ids, joint_names = self._asset.find_joints_by_actuator_names(
      cfg.actuator_names
    )
    self._joint_ids = torch.tensor(joint_ids, device=self.device, dtype=torch.long)
    self._joint_names = joint_names

    self._num_joints = len(joint_ids)
    self._action_dim = len(joint_ids)

    self._raw_actions = torch.zeros(self.num_envs, self.action_dim, device=self.device)
    self._processed_actions = torch.zeros_like(self._raw_actions)

    if isinstance(cfg.scale, (float, int)):
      self._scale = float(cfg.scale)
    elif isinstance(cfg.scale, dict):
      self._scale = torch.ones(self.num_envs, self.action_dim, device=self.device)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.scale, self._joint_names
      )
      self._scale[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported scale type: {type(cfg.scale)}."
        " Supported types are float and dict."
      )

    if isinstance(cfg.offset, (float, int)):
      self._offset = float(cfg.offset)
    elif isinstance(cfg.offset, dict):
      self._offset = torch.zeros_like(self._raw_actions)
      index_list, _, value_list = resolve_matching_names_values(
        cfg.offset, self._joint_names
      )
      self._offset[:, index_list] = torch.tensor(value_list, device=self.device)
    else:
      raise ValueError(
        f"Unsupported offset type: {type(cfg.offset)}."
        " Supported types are float and dict."
      )

  # Properties.

  @property
  def scale(self) -> torch.Tensor | float:
    return self._scale

  @property
  def offset(self) -> torch.Tensor | float:
    return self._offset

  @property
  def raw_action(self) -> torch.Tensor:
    return self._raw_actions

  @property
  def action_dim(self) -> int:
    return self._action_dim

  def process_actions(self, actions: torch.Tensor):
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset

  def reset(self, env_ids: torch.Tensor | slice | None = None) -> None:
    self._raw_actions[env_ids] = 0.0


class JointPositionAction(JointAction):
  def __init__(
    self, cfg: actions_config.JointPositionActionCfg, env: ManagerBasedRlEnv
  ):
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

  def apply_actions(self) -> None:
    self._asset.set_joint_position_target(
      self._processed_actions, joint_ids=self._joint_ids
    )


class JointVelocityAction(JointAction):
  def __init__(
    self, cfg: actions_config.JointVelocityActionCfg, env: ManagerBasedRlEnv
  ):
    super().__init__(cfg=cfg, env=env)

    if cfg.use_default_offset:
      self._offset = self._asset.data.default_joint_vel[:, self._joint_ids].clone()

  def apply_actions(self) -> None:
    self._asset.set_joint_velocity_target(
      self._processed_actions, joint_ids=self._joint_ids
    )


class JointEffortAction(JointAction):
  def __init__(self, cfg: actions_config.JointEffortActionCfg, env: ManagerBasedRlEnv):
    super().__init__(cfg=cfg, env=env)

  def apply_actions(self) -> None:
    self._asset.set_joint_effort_target(
      self._processed_actions, joint_ids=self._joint_ids
    )


class MotionTrackingJointPositionAction(JointAction):
  def __init__(
    self,
    cfg: actions_config.MotionTrackingJointPositionActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg=cfg, env=env)

    # Get motion command from command manager
    # Default command name is "motion" (can be made configurable if needed)
    command_name = getattr(cfg, "command_name", "motion")

    # Check if command manager exists and has the motion command
    if hasattr(env, "command_manager") and env.command_manager is not None:
      try:
        from mjlab.tasks.tracking.mdp.commands import MotionCommand

        motion_command: MotionCommand = cast(
          MotionCommand, env.command_manager.get_term(command_name)
        )
        self._motion_command = motion_command
      except (KeyError, AttributeError):
        # Fallback to default offset if motion command not available
        self._motion_command = None
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
    else:
      # Fallback to default offset if command manager not available
      self._motion_command = None
      self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

  def process_actions(self, actions: torch.Tensor):
    # Update offset from motion tracking command if available
    if self._motion_command is not None:
      # Get current joint positions from motion command
      motion_joint_pos = self._motion_command.joint_pos  # (num_envs, num_joints)
      # Select the joints that this action uses
      self._offset = motion_joint_pos[:, self._joint_ids].clone()

    # Process actions with updated offset
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset

  def apply_actions(self):
    self._asset.set_joint_position_target(
      self._processed_actions, joint_ids=self._joint_ids
    )


class MotionTrackingPDTargetsAction(MotionTrackingJointPositionAction):
  def __init__(
    self,
    cfg: actions_config.MotionTrackingPDTargetsActionCfg,
    env: ManagerBasedRlEnv,
  ):
    super().__init__(cfg=cfg, env=env)

    command_name = getattr(cfg, "command_name", "motion")

    # Check if command manager exists and has the motion command
    if hasattr(env, "command_manager") and env.command_manager is not None:
      try:
        from mjlab.tasks.tracking.mdp.commands import MotionCommand

        motion_command: MotionCommand = cast(
          MotionCommand, env.command_manager.get_term(command_name)
        )
        self._motion_command = motion_command
      except (KeyError, AttributeError):
        # Fallback to default offset if motion command not available
        self._motion_command = None
        self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()
    else:
      # Fallback to default offset if command manager not available
      self._motion_command = None
      self._offset = self._asset.data.default_joint_pos[:, self._joint_ids].clone()

  def process_actions(self, actions: torch.Tensor):
    # Update offset from motion tracking command if available
    if self._motion_command is not None:
      # Get current joint positions from motion command
      motion_pd_targets = (
        self._motion_command.joint_pd_targets
      )  # (num_envs, num_joints)
      if motion_pd_targets is None:
        raise ValueError("PD targets not available")
      # Select the joints that this action uses
      self._offset = motion_pd_targets[:, self._joint_ids]

    # Process actions with updated offset
    self._raw_actions[:] = actions
    self._processed_actions = self._raw_actions * self._scale + self._offset
