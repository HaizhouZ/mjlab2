from __future__ import annotations

from typing import TYPE_CHECKING, cast

import torch

from mjlab.entity import Entity
from mjlab.managers.scene_entity_config import SceneEntityCfg
from mjlab.utils.lab_api.math import (
  matrix_from_quat,
  subtract_frame_transforms,
)

from .commands import MotionCommand, MultiMotionCommand

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


def _get_horizon(command) -> int:
  """Get horizon from command if available."""
  if isinstance(command, MultiMotionCommand):
    return command.cfg.horizon
  return 0


def _flatten_horizon(data: torch.Tensor) -> torch.Tensor:
  """Flatten horizon dimension if present.

  Args:
    data: Tensor of shape (num_envs, ...) or (num_envs, horizon, ...)

  Returns:
    Tensor with horizon dimension flattened: (num_envs, horizon * ...) or (num_envs, ...)
  """
  if data.ndim > 2 and data.shape[1] > 1:
    # Has horizon dimension, flatten it
    return data.reshape(data.shape[0], -1)
  return data.reshape(data.shape[0], -1)


def motion_anchor_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  # Handle both MotionCommand and MultiMotionCommand
  if not isinstance(command, (MotionCommand, MultiMotionCommand)):
    raise TypeError(
      f"Expected MotionCommand or MultiMotionCommand, got {type(command)}"
    )

  horizon = 1
  robot_anchor_pos = command.robot_anchor_pos_w  # (num_envs, 3) - actual robot state
  robot_anchor_quat = command.robot_anchor_quat_w  # (num_envs, 4) - actual robot state

  # Use horizon-aware methods if horizon > 0, otherwise use properties
  if horizon > 0 and isinstance(command, MultiMotionCommand):
    anchor_pos = command.get_anchor_pos_w_horizon(horizon)  # (num_envs, horizon, 3)
    anchor_quat = command.get_anchor_quat_w_horizon(horizon)  # (num_envs, horizon, 4)
  else:
    anchor_pos = command.anchor_pos_w  # (num_envs, 3)
    anchor_quat = command.anchor_quat_w  # (num_envs, 4)

  # Expand robot state to match motion data shape dynamically
  if anchor_pos.ndim == 3:
    # Has horizon: (num_envs, horizon, 3)
    robot_anchor_pos = robot_anchor_pos[:, None, :].repeat(1, anchor_pos.shape[1], 1)
    robot_anchor_quat = robot_anchor_quat[:, None, :].repeat(1, anchor_pos.shape[1], 1)

  pos, _ = subtract_frame_transforms(
    robot_anchor_pos,
    robot_anchor_quat,
    anchor_pos,
    anchor_quat,
  )

  return _flatten_horizon(pos)


def motion_anchor_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (MotionCommand, MultiMotionCommand)):
    raise TypeError(
      f"Expected MotionCommand or MultiMotionCommand, got {type(command)}"
    )

  horizon = 1
  robot_anchor_pos = command.robot_anchor_pos_w  # (num_envs, 3) - actual robot state
  robot_anchor_quat = command.robot_anchor_quat_w  # (num_envs, 4) - actual robot state

  # Use horizon-aware methods if horizon > 0, otherwise use properties
  if horizon > 0 and isinstance(command, MultiMotionCommand):
    anchor_pos = command.get_anchor_pos_w_horizon(horizon)  # (num_envs, horizon, 3)
    anchor_quat = command.get_anchor_quat_w_horizon(horizon)  # (num_envs, horizon, 4)
  else:
    anchor_pos = command.anchor_pos_w  # (num_envs, 3)
    anchor_quat = command.anchor_quat_w  # (num_envs, 4)

  # Expand robot state to match motion data shape dynamically
  if anchor_pos.ndim == 3:
    # Has horizon: (num_envs, horizon, 3)
    robot_anchor_pos = robot_anchor_pos[:, None, :].repeat(1, anchor_pos.shape[1], 1)
    robot_anchor_quat = robot_anchor_quat[:, None, :].repeat(1, anchor_pos.shape[1], 1)

  _, ori = subtract_frame_transforms(
    robot_anchor_pos,
    robot_anchor_quat,
    anchor_pos,
    anchor_quat,
  )
  mat = matrix_from_quat(ori)
  # Extract first 2 rows of rotation matrix and flatten
  mat_2d = mat[..., :2]  # (num_envs, horizon?, 2, 3) or (num_envs, 2, 3)
  return _flatten_horizon(mat_2d)


def robot_body_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (MotionCommand, MultiMotionCommand)):
    raise TypeError(
      f"Expected MotionCommand or MultiMotionCommand, got {type(command)}"
    )

  horizon = 1
  num_bodies = len(command.cfg.body_names)
  robot_anchor_pos = command.robot_anchor_pos_w  # (num_envs, 3) - actual robot state
  robot_anchor_quat = command.robot_anchor_quat_w  # (num_envs, 4) - actual robot state

  # Use horizon-aware methods if horizon > 0, otherwise use properties
  if horizon > 0 and isinstance(command, MultiMotionCommand):
    body_pos_w = command.get_body_pos_w_horizon(
      horizon
    )  # (num_envs, horizon, num_bodies, 3)
    body_quat_w = command.get_body_quat_w_horizon(
      horizon
    )  # (num_envs, horizon, num_bodies, 4)
  else:
    body_pos_w = command.body_pos_w  # (num_envs, num_bodies, 3)
    body_quat_w = command.body_quat_w  # (num_envs, num_bodies, 4)

  # Expand robot anchor to match motion data shape dynamically
  # body_pos_w can be (num_envs, num_bodies, 3) or (num_envs, horizon, num_bodies, 3)
  if body_pos_w.ndim == 4:
    # Has horizon: (num_envs, horizon, num_bodies, 3)
    robot_anchor_pos_expanded = robot_anchor_pos[:, None, None, :].repeat(
      1, body_pos_w.shape[1], num_bodies, 1
    )
    robot_anchor_quat_expanded = robot_anchor_quat[:, None, None, :].repeat(
      1, body_pos_w.shape[1], num_bodies, 1
    )
  else:
    # No horizon: (num_envs, num_bodies, 3)
    robot_anchor_pos_expanded = robot_anchor_pos[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_quat_expanded = robot_anchor_quat[:, None, :].repeat(1, num_bodies, 1)

  pos_b, _ = subtract_frame_transforms(
    robot_anchor_pos_expanded,
    robot_anchor_quat_expanded,
    body_pos_w,
    body_quat_w,
  )

  return _flatten_horizon(pos_b)


def robot_body_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (MotionCommand, MultiMotionCommand)):
    raise TypeError(
      f"Expected MotionCommand or MultiMotionCommand, got {type(command)}"
    )

  horizon = 1
  num_bodies = len(command.cfg.body_names)
  robot_anchor_pos = command.robot_anchor_pos_w  # (num_envs, 3) - actual robot state
  robot_anchor_quat = command.robot_anchor_quat_w  # (num_envs, 4) - actual robot state

  # Use horizon-aware methods if horizon > 0, otherwise use properties
  if horizon > 0 and isinstance(command, MultiMotionCommand):
    body_pos_w = command.get_body_pos_w_horizon(
      horizon
    )  # (num_envs, horizon, num_bodies, 3)
    body_quat_w = command.get_body_quat_w_horizon(
      horizon
    )  # (num_envs, horizon, num_bodies, 4)
  else:
    body_pos_w = command.body_pos_w  # (num_envs, num_bodies, 3)
    body_quat_w = command.body_quat_w  # (num_envs, num_bodies, 4)

  # Expand robot anchor to match motion data shape dynamically
  # body_pos_w can be (num_envs, num_bodies, 3) or (num_envs, horizon, num_bodies, 3)
  if body_pos_w.ndim == 4:
    # Has horizon: (num_envs, horizon, num_bodies, 3)
    robot_anchor_pos_expanded = robot_anchor_pos[:, None, None, :].repeat(
      1, body_pos_w.shape[1], num_bodies, 1
    )
    robot_anchor_quat_expanded = robot_anchor_quat[:, None, None, :].repeat(
      1, body_pos_w.shape[1], num_bodies, 1
    )
  else:
    # No horizon: (num_envs, num_bodies, 3)
    robot_anchor_pos_expanded = robot_anchor_pos[:, None, :].repeat(1, num_bodies, 1)
    robot_anchor_quat_expanded = robot_anchor_quat[:, None, :].repeat(1, num_bodies, 1)

  _, ori_b = subtract_frame_transforms(
    robot_anchor_pos_expanded,
    robot_anchor_quat_expanded,
    body_pos_w,
    body_quat_w,
  )
  mat = matrix_from_quat(ori_b)
  # Extract first 2 rows of rotation matrix and flatten
  mat_2d = mat[
    ..., :2
  ]  # (num_envs, horizon?, num_bodies, 2, 3) or (num_envs, num_bodies, 2, 3)
  return _flatten_horizon(mat_2d)


def object_pos_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (MotionCommand, MultiMotionCommand)):
    raise TypeError(
      f"Expected MotionCommand or MultiMotionCommand, got {type(command)}"
    )

  horizon = 1
  robot_anchor_pos = command.robot_anchor_pos_w  # (num_envs, 3) - actual robot state
  robot_anchor_quat = command.robot_anchor_quat_w  # (num_envs, 4) - actual robot state

  # Use horizon-aware methods if horizon > 0, otherwise use properties
  if horizon > 0 and isinstance(command, MultiMotionCommand):
    object_pos = command.get_object_pos_w_horizon(horizon)  # (num_envs, horizon, 3)
    object_quat = command.get_object_quat_w_horizon(horizon)  # (num_envs, horizon, 4)
  else:
    object_pos = command.object_pos_w  # (num_envs, 3)
    object_quat = command.object_quat_w  # (num_envs, 4)

  # Expand robot state to match motion data shape dynamically
  if object_pos.ndim == 3:
    # Has horizon: (num_envs, horizon, 3)
    robot_anchor_pos = robot_anchor_pos[:, None, :].repeat(1, object_pos.shape[1], 1)
    robot_anchor_quat = robot_anchor_quat[:, None, :].repeat(1, object_pos.shape[1], 1)

  pos, _ = subtract_frame_transforms(
    robot_anchor_pos,
    robot_anchor_quat,
    object_pos,
    object_quat,
  )
  # breakpoint()
  return _flatten_horizon(pos)


def object_lin_vel_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  object: Entity = env.scene[asset_cfg.name]
  return object.data.body_link_lin_vel_w[:, 0].view(env.num_envs, -1)


def object_ang_vel_w(env: ManagerBasedRlEnv, asset_cfg: SceneEntityCfg) -> torch.Tensor:
  object: Entity = env.scene[asset_cfg.name]
  return object.data.body_link_ang_vel_w[:, 0].view(env.num_envs, -1)


def object_position_error(
  env: ManagerBasedRlEnv, command_name: str, asset_cfg: SceneEntityCfg
) -> torch.Tensor:
  """Compute the error between actual and desired object position.

  This is crucial for object pushing - the robot needs to know how far the object
  is from where it should be to learn to push it correctly.
  """

  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (MotionCommand, MultiMotionCommand)):
    raise TypeError(
      f"Expected MotionCommand or MultiMotionCommand, got {type(command)}"
    )
  object: Entity = env.scene[asset_cfg.name]

  horizon = 1

  # Use horizon-aware methods if horizon > 0, otherwise use properties
  if horizon > 0 and isinstance(command, MultiMotionCommand):
    desired_pos = command.get_object_pos_w_horizon(horizon)  # (num_envs, horizon, 3)
  else:
    desired_pos = command.object_pos_w  # (num_envs, 3)

  # Actual object position from simulation
  actual_pos = object.data.body_link_pos_w[:, 0]  # (num_envs, 3)

  # Expand actual_pos to match desired_pos shape dynamically
  if desired_pos.ndim == 3:
    # Has horizon: (num_envs, horizon, 3)
    actual_pos = actual_pos[:, None, :].repeat(1, desired_pos.shape[1], 1)

  error = actual_pos - desired_pos
  return _flatten_horizon(error)


def object_orientation_error(
  env: ManagerBasedRlEnv,
  command_name: str,
  asset_cfg: SceneEntityCfg,
) -> torch.Tensor:
  """Compute the error between actual and desired object orientation."""

  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (MotionCommand, MultiMotionCommand)):
    raise TypeError(
      f"Expected MotionCommand or MultiMotionCommand, got {type(command)}"
    )
  object: Entity = env.scene[asset_cfg.name]

  horizon = 1

  # Use horizon-aware methods if horizon > 0, otherwise use properties
  if horizon > 0 and isinstance(command, MultiMotionCommand):
    object_pos = command.get_object_pos_w_horizon(horizon)  # (num_envs, horizon, 3)
    object_quat = command.get_object_quat_w_horizon(horizon)  # (num_envs, horizon, 4)
  else:
    object_pos = command.object_pos_w  # (num_envs, 3)
    object_quat = command.object_quat_w  # (num_envs, 4)

  # Actual object pose from simulation
  actual_pos = object.data.body_link_pos_w[:, 0]  # (num_envs, 3)
  actual_quat = object.data.body_link_quat_w[:, 0]  # (num_envs, 4)

  # Expand actual pose to match motion data shape dynamically
  if object_pos.ndim == 3:
    # Has horizon: (num_envs, horizon, 3)
    actual_pos = actual_pos[:, None, :].repeat(1, object_pos.shape[1], 1)
    actual_quat = actual_quat[:, None, :].repeat(1, object_pos.shape[1], 1)

  _, ori = subtract_frame_transforms(
    object_pos,
    object_quat,
    actual_pos,
    actual_quat,
  )

  mat = matrix_from_quat(ori)
  # Extract first 2 rows of rotation matrix and flatten
  mat_2d = mat[..., :2]  # (num_envs, horizon?, 2, 3) or (num_envs, 2, 3)
  # breakpoint()
  return _flatten_horizon(mat_2d)


def object_ori_b(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (MotionCommand, MultiMotionCommand)):
    raise TypeError(
      f"Expected MotionCommand or MultiMotionCommand, got {type(command)}"
    )

  horizon = 1
  robot_anchor_pos = command.robot_anchor_pos_w  # (num_envs, 3) - actual robot state
  robot_anchor_quat = command.robot_anchor_quat_w  # (num_envs, 4) - actual robot state

  # Use horizon-aware methods if horizon > 0, otherwise use properties
  if horizon > 0 and isinstance(command, MultiMotionCommand):
    object_pos = command.get_object_pos_w_horizon(horizon)  # (num_envs, horizon, 3)
    object_quat = command.get_object_quat_w_horizon(horizon)  # (num_envs, horizon, 4)
  else:
    object_pos = command.object_pos_w  # (num_envs, 3)
    object_quat = command.object_quat_w  # (num_envs, 4)

  # Expand robot state to match motion data shape dynamically
  if object_pos.ndim == 3:
    # Has horizon: (num_envs, horizon, 3)
    robot_anchor_pos = robot_anchor_pos[:, None, :].repeat(1, object_pos.shape[1], 1)
    robot_anchor_quat = robot_anchor_quat[:, None, :].repeat(1, object_pos.shape[1], 1)

  _, ori = subtract_frame_transforms(
    robot_anchor_pos,
    robot_anchor_quat,
    object_pos,
    object_quat,
  )
  mat = matrix_from_quat(ori)
  # Extract first 2 rows of rotation matrix and flatten
  mat_2d = mat[..., :2]  # (num_envs, horizon?, 2, 3) or (num_envs, 2, 3)
  # breakpoint()
  return _flatten_horizon(mat_2d)


def contact_indicator(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  command = env.command_manager.get_term(command_name)
  if not isinstance(command, (MotionCommand, MultiMotionCommand)):
    raise TypeError(
      f"Expected MotionCommand or MultiMotionCommand, got {type(command)}"
    )
  if command.ref_object_contact is None:
    # Return zeros if no contact data available
    return torch.zeros(env.num_envs, 1, device=env.device)
  return command.ref_object_contact.view(env.num_envs, -1)


def trajectory_encoding(env: ManagerBasedRlEnv, command_name: str) -> torch.Tensor:
  """Encode a trajectory with horizon from cfg.

  The trajectory contains:
    - joint_pos (29)
    - joint_vel (29)
    - anchor_pos (3)
    - anchor_quat (4)
    - object_pos (3)
    - object_quat (4)
  Total: 72 features per timestep

  Args:
    env: The environment instance
    command_name: Name of the command term to use

  Returns:
    Flattened trajectory tensor of shape (num_envs, horizon * 72)
  """
  command = cast(MultiMotionCommand, env.command_manager.get_term(command_name))
  horizon = _get_horizon(command)

  # Get trajectory data with horizon
  joint_pos = command.get_joint_pos_horizon(horizon)  # (num_envs, horizon, 29)
  joint_vel = command.get_joint_vel_horizon(horizon)  # (num_envs, horizon, 29)
  anchor_pos = command.get_anchor_pos_w_horizon(horizon)  # (num_envs, horizon, 3)
  anchor_quat = command.get_anchor_quat_w_horizon(horizon)  # (num_envs, horizon, 4)
  # Concatenate along feature dimension: (num_envs, horizon, 72)
  trajectory = torch.cat(
    [
      joint_pos,  # (num_envs, horizon, 29)
      joint_vel,  # (num_envs, horizon, 29)
      anchor_pos,  # (num_envs, horizon, 3)
      anchor_quat,  # (num_envs, horizon, 4)
    ],
    dim=2,
  )
  if command.trajectory_encoder is not None:
    trajectory = command.trajectory_encoder(trajectory)
  # Flatten horizon dimension: (num_envs, horizon * 72)
  return trajectory.reshape(env.num_envs, -1)
