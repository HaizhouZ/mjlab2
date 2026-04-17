import os
from typing import cast

import torch
import torch.nn as nn

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.exporter_utils import (
  attach_metadata_to_onnx,
  get_base_metadata,
)
from mjlab.tasks.tracking.mdp import MotionCommand, MultiMotionCommand
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


def get_input_dim(model: nn.Module) -> int:
  if hasattr(model, "input_size"):
    return model.input_size  # type: ignore

  if hasattr(model, "actor_obs_dim"):
    return model.actor_obs_dim  # type: ignore

  if hasattr(model, "obs_dim"):
    return model.obs_dim  # type: ignore

  if hasattr(model, "actor_mean"):
    return get_input_dim(model.actor_mean)  # type: ignore

  if hasattr(model, "actor"):
    return get_input_dim(model.actor)  # type: ignore

  if hasattr(model, "mlp"):
    return get_input_dim(model.mlp)  # type: ignore

  # Get the very first module in the Sequential
  first = model[0] if isinstance(model, nn.Sequential) else model
  # 1. Check for Linear
  if hasattr(first, "in_features"):
    return first.in_features  # type: ignore

  # 2. Check for LayerNorm
  if hasattr(first, "normalized_shape"):
    return first.normalized_shape[0]  # type: ignore

  # 3. Check for custom blocks (like your residual_block)
  if hasattr(first, "input_dim"):  # If you saved it as an attribute
    return first.input_dim  # type: ignore

  raise AttributeError("Could not automatically determine input dimension.")


class _OnnxMotionPolicyExporter(_OnnxPolicyExporter):
  def __init__(
    self, env: ManagerBasedRlEnv, actor_critic, normalizer=None, verbose=False
  ):
    super().__init__(actor_critic, normalizer, verbose)
    self.input_dim = get_input_dim(actor_critic)
    cmd = env.command_manager.get_term("motion")

    # Handle both MotionCommand and MultiMotionCommand
    if isinstance(cmd, MultiMotionCommand):
      # For MultiMotionCommand, use the first motion from the loader
      # Access the actual motion data (not padded) from the first motion
      first_motion = cmd.motion_loader.motions[0]
      self.register_buffer("joint_pos", first_motion.joint_pos.to("cpu"))
      self.register_buffer("joint_vel", first_motion.joint_vel.to("cpu"))
      self.register_buffer("body_pos_w", first_motion.body_pos_w.to("cpu"))
      self.register_buffer("body_quat_w", first_motion.body_quat_w.to("cpu"))
      self.register_buffer("body_lin_vel_w", first_motion.body_lin_vel_w.to("cpu"))
      self.register_buffer("body_ang_vel_w", first_motion.body_ang_vel_w.to("cpu"))

    else:
      # Original MotionCommand handling
      cmd = cast(MotionCommand, cmd)
      self.register_buffer("joint_pos", cmd.motion.joint_pos.to("cpu"))
      self.register_buffer("joint_vel", cmd.motion.joint_vel.to("cpu"))
      self.register_buffer("body_pos_w", cmd.motion.body_pos_w.to("cpu"))
      self.register_buffer("body_quat_w", cmd.motion.body_quat_w.to("cpu"))
      self.register_buffer("body_lin_vel_w", cmd.motion.body_lin_vel_w.to("cpu"))
      self.register_buffer("body_ang_vel_w", cmd.motion.body_ang_vel_w.to("cpu"))

    self.time_step_total: int = self.joint_pos.shape[0]

  def forward(self, x, time_step):  # type: ignore[invalid-method-override]
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

    return outputs

  def export(self, path, filename):
    export_device = next(self.actor.parameters()).device
    self.to(export_device)
    obs = torch.zeros(1, self.input_dim, device=export_device)
    time_step = torch.zeros(1, 1, device=export_device)

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

    torch.onnx.export(
      self,
      (obs, time_step),
      os.path.join(path, filename),
      export_params=True,
      opset_version=18,
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
  # Handle both MotionCommand and MultiMotionCommand
  assert isinstance(motion_term, (MotionCommand, MultiMotionCommand))
  motion_term_cfg = motion_term.cfg
  metadata.update(
    {
      "anchor_body_name": motion_term_cfg.anchor_body_name,
      "body_names": list(motion_term_cfg.body_names),
    }
  )
  attach_metadata_to_onnx(onnx_path, metadata)
