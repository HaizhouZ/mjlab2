import os
from types import SimpleNamespace

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


def _get_exportable_policy(policy):
  if hasattr(policy, "as_onnx"):
    return SimpleNamespace(actor=policy.as_onnx(False), is_recurrent=False)
  if hasattr(policy, "actor") or hasattr(policy, "student"):
    return policy
  return SimpleNamespace(
    actor=policy,
    is_recurrent=getattr(policy, "is_recurrent", False),
    memory_a=getattr(policy, "memory_a", None),
    memory_s=getattr(policy, "memory_s", None),
  )


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

    policy_path = path.split("model")[0]
    filename = policy_path.split("/")[-2] + ".onnx"
    policy = self.alg.policy
    normalizer = _get_policy_normalizer(policy)
    export_motion_policy_as_onnx(
      self.env.unwrapped,
      _get_exportable_policy(policy),
      normalizer=normalizer,
      path=policy_path,
      filename=filename,
    )
    # Attach metadata (use "local" for run_path if not using wandb)
    run_name = (
      wandb.run.name if self.logger.logger_type == "wandb" and wandb.run else "local"
    )
    attach_onnx_metadata(
      self.env.unwrapped,
      run_name,  # type: ignore
      path=policy_path,
      filename=filename,
    )
    if self.logger.logger_type in ["wandb"]:
      wandb.save(policy_path + filename, base_path=os.path.dirname(policy_path))
      # link the artifact registry to this run
      if self.registry_name is not None:
        wandb.run.use_artifact(self.registry_name)  # type: ignore
        self.registry_name = None
