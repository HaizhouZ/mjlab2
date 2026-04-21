import os
from types import SimpleNamespace

import torch
import torch.nn as nn
import wandb
from rsl_rl.env.vec_env import VecEnv

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.tracking.rl.exporter import (
  attach_onnx_metadata,
  export_motion_policy_as_onnx,
)


def _get_policy_normalizer(policy):
  if getattr(policy, "actor_obs_normalization", False):
    return getattr(policy, "actor_obs_normalizer", None)
  if getattr(policy, "obs_normalization", False):
    return getattr(policy, "obs_normalizer", None)
  return None


class _InferenceActorWrapper(nn.Module):
  def __init__(self, policy: nn.Module):
    super().__init__()
    self.policy = policy
    self.input_dim = getattr(policy, "actor_obs_dim", getattr(policy, "obs_dim", None))

  def forward(self, obs: torch.Tensor) -> torch.Tensor:
    return self.policy.act_inference(obs)


def _get_exportable_policy(policy):
  if hasattr(policy, "as_onnx"):
    return SimpleNamespace(actor=policy.as_onnx(False), is_recurrent=False)
  if hasattr(policy, "act_inference") and hasattr(policy, "actor"):
    return SimpleNamespace(
      actor=_InferenceActorWrapper(policy),
      is_recurrent=getattr(policy, "is_recurrent", False),
      memory_a=getattr(policy, "memory_a", None),
      memory_s=getattr(policy, "memory_s", None),
    )
  if hasattr(policy, "actor") or hasattr(policy, "student"):
    return policy
  return SimpleNamespace(
    actor=policy,
    is_recurrent=getattr(policy, "is_recurrent", False),
    memory_a=getattr(policy, "memory_a", None),
    memory_s=getattr(policy, "memory_s", None),
  )


def _save_tracking_policy(runner, path: str) -> None:
  policy_path = path.split("model")[0]
  filename = policy_path.split("/")[-2] + ".onnx"
  policy = runner.alg.policy
  normalizer = _get_policy_normalizer(policy)
  logger_type = getattr(getattr(runner, "logger", None), "logger_type", None)
  export_motion_policy_as_onnx(
    runner.env.unwrapped,
    _get_exportable_policy(policy),
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
    if getattr(runner, "registry_name", None) is not None and wandb.run is not None:
      wandb.run.use_artifact(runner.registry_name)  # type: ignore[arg-type]
      runner.registry_name = None


class MotionTrackingOnPolicyRunner(MjlabOnPolicyRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def save(self, path: str, infos=None):
    """Save the model and training information."""
    super().save(path, infos)
    _save_tracking_policy(self, path)
