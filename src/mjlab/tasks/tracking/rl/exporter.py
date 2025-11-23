import os
from typing import cast

import torch

from mjlab.envs import ManagerBasedRlEnv
from mjlab.envs.mdp.actions import (
  JointPositionActionCfg,
  MotionTrackingJointPositionActionCfg,
)
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.tasks.tracking.mdp import MotionCommand
from mjlab.utils.lab_api.rl.exporter import _OnnxPolicyExporter


def export_motion_policy_as_onnx(
  env: ManagerBasedRlEnv,
  actor_critic: object,
  path: str,
  normalizer: object | None = None,
  filename="policy.onnx",
  verbose=False,
):
  if not os.path.exists(path):
    os.makedirs(path, exist_ok=True)
  policy_exporter = _OnnxMotionPolicyExporter(env, actor_critic, normalizer, verbose)
  policy_exporter.export(path, filename)


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
  def __init__(
    self, env: ManagerBasedRlEnv, actor_critic, normalizer=None, verbose=False
  ):
    super().__init__(actor_critic, normalizer, verbose)
    cmd = cast(MotionCommand, env.command_manager.get_term("motion"))

    self.joint_pos = cmd.motion.joint_pos.to("cpu")
    self.joint_vel = cmd.motion.joint_vel.to("cpu")
    self.body_pos_w = cmd.motion.body_pos_w.to("cpu")
    self.body_quat_w = cmd.motion.body_quat_w.to("cpu")
    self.body_lin_vel_w = cmd.motion.body_lin_vel_w.to("cpu")
    self.body_ang_vel_w = cmd.motion.body_ang_vel_w.to("cpu")

    self.has_object_data = hasattr(cmd.motion, "_object_pos_w")
    if self.has_object_data:
      self.object_pos_w = cmd.motion.object_pos_w.to("cpu")
      self.object_quat_w = cmd.motion.object_quat_w.to("cpu")
      object_lin_vel_w = cmd.motion.object_lin_vel_w
      self.object_lin_vel_w = (
        object_lin_vel_w.to("cpu") if object_lin_vel_w is not None else None
      )
      object_ang_vel_w = cmd.motion.object_ang_vel_w
      self.object_ang_vel_w = (
        object_ang_vel_w.to("cpu") if object_ang_vel_w is not None else None
      )
      object_contact = cmd.motion.object_contact
      self.contact_indicators = (
        object_contact.to("cpu") if object_contact is not None else None
      )
      contact_positions = cmd.motion.contact_positions
      self.contact_positions = (
        contact_positions.to("cpu") if contact_positions is not None else None
      )

    self.time_step_total: int = self.joint_pos.shape[0]

  def forward(self, x, time_step):  # pyright: ignore [reportIncompatibleMethodOverride]
    time_step_clamped = torch.clamp(
      time_step.long().squeeze(-1), max=self.time_step_total - 1
    )
    outputs = (
      self.actor(self.normalizer(x)),
      self.joint_pos[time_step_clamped],
      self.joint_vel[time_step_clamped],
      self.body_pos_w[time_step_clamped],
      self.body_quat_w[time_step_clamped],
      self.body_lin_vel_w[time_step_clamped],
      self.body_ang_vel_w[time_step_clamped],
    )

    # Only include object outputs if object data exists
    if self.has_object_data:
      object_outputs = (
        self.object_pos_w[time_step_clamped],
        self.object_quat_w[time_step_clamped],
        self.object_lin_vel_w[time_step_clamped]
        if self.object_lin_vel_w is not None
        else torch.zeros(len(time_step_clamped), 3),
        self.object_ang_vel_w[time_step_clamped]
        if self.object_ang_vel_w is not None
        else torch.zeros(len(time_step_clamped), 3),
        self.contact_indicators[time_step_clamped]
        if self.contact_indicators is not None
        else torch.zeros(len(time_step_clamped), dtype=torch.bool),
        self.contact_positions[time_step_clamped]
        if self.contact_positions is not None
        else torch.zeros(len(time_step_clamped), 2, 3),
      )
      return outputs + object_outputs
    else:
      return outputs

  def export(self, path, filename):
    self.to("cpu")
    obs = torch.zeros(1, self.actor[0].in_features)
    time_step = torch.zeros(1, 1)

    # Base output names (always included)
    output_names = [
      "actions",
      "joint_pos",
      "joint_vel",
      "body_pos_w",
      "body_quat_w",
      "body_lin_vel_w",
      "body_ang_vel_w",
    ]

    # Only include object outputs if object data exists
    if self.has_object_data:
      output_names.extend(
        [
          "object_pos_w",
          "object_quat_w",
          "object_lin_vel_w",
          "object_ang_vel_w",
          "contact_indicators",
          "contact_positions",
        ]
      )

    torch.onnx.export(
      self,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=11,
      verbose=self.verbose,
      input_names=["obs", "time_step"],
      output_names=output_names,
      dynamic_axes={},
      dynamo=False,
    )


def attach_onnx_metadata(
  env: ManagerBasedRlEnv, run_path: str, path: str, filename="policy.onnx"
) -> None:
  """Attach tracking-specific metadata to ONNX model.

  Args:
    env: The RL environment.
    run_path: W&B run path or other identifier.
    path: Directory containing the ONNX file.
    filename: Name of the ONNX file.
  """
  onnx_path = os.path.join(path, filename)

  # Get base metadata common to all tasks.
  metadata = get_base_metadata(env, run_path)

  # Add tracking-specific metadata.
  motion_term = env.command_manager.get_term("motion")
  assert isinstance(motion_term, MotionCommand)
  motion_term_cfg = motion_term.cfg

  # Determine use_motion_offset based on action configuration
  joint_pos_action = env.cfg.actions.get("joint_pos")
  if isinstance(joint_pos_action, MotionTrackingJointPositionActionCfg):
    use_motion_offset = True
  elif isinstance(joint_pos_action, JointPositionActionCfg):
    use_motion_offset = False
  else:
    # Default to False if action type is unknown
    use_motion_offset = False

  metadata.update(
    {
      "anchor_body_name": motion_term_cfg.anchor_body_name,
      "body_names": list(motion_term_cfg.body_names),
      "use_motion_offset": use_motion_offset,
    }
  )
  attach_metadata_to_onnx(onnx_path, metadata)
