import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import wandb
from rsl_rl.runners import ReppoRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)


class _InferenceActorWrapper(nn.Module):
  def __init__(self, policy: nn.Module):
    super().__init__()
    self.policy = policy
    self.input_dim = getattr(
      policy, "actor_obs_dim", getattr(policy, "obs_dim", None)
    )

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.policy.act_inference(obs)


def _get_exportable_policy(policy):
  if hasattr(policy, "as_onnx"):
    return policy
  if hasattr(policy, "act_inference") and hasattr(policy, "actor"):
    return SimpleNamespace(
      actor=_InferenceActorWrapper(policy),
      is_recurrent=getattr(policy, "is_recurrent", False),
    )
  return policy


def _export_velocity_policy(runner, path: str) -> None:
  policy_path = path.split("model")[0]
  filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
  if runner.alg.policy.actor_obs_normalization:
    normalizer = runner.alg.policy.actor_obs_normalizer
  else:
    normalizer = None
  logger_type = getattr(getattr(runner, "logger", None), "logger_type", None)
  export_velocity_policy_as_onnx(
    _get_exportable_policy(runner.alg.policy),
    normalizer=normalizer,
    path=policy_path,
    filename=filename,
  )
  run_name = wandb.run.name if logger_type == "wandb" and wandb.run else "local"
  attach_onnx_metadata(
    runner.env.unwrapped,
    run_name,  # type: ignore[arg-type]
    path=policy_path,
    filename=filename,
  )
  if logger_type in ["wandb"]:
    wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))


class VelocityOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    _export_velocity_policy(self, path)


class VelocityReppoRunner(ReppoRunner):
  env: RslRlVecEnvWrapper

  def save(self, path: str, infos=None):
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    super().save(path, {**(infos or {}), "env_state": env_state})
    _export_velocity_policy(self, path)

  def load(
    self, path: str, load_optimizer: bool = True, map_location: str | None = None
  ):
    infos = super().load(
      path, load_optimizer=load_optimizer, map_location=map_location
    )
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"][
        "common_step_counter"
      ]
    return infos
