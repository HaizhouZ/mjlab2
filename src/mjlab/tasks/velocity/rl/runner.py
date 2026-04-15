import os

import wandb
from rsl_rl.runners import ReppoRunner

from mjlab.rl import RslRlVecEnvWrapper
from mjlab.rl.runner import MjlabOnPolicyRunner
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata,
  export_velocity_policy_as_onnx,
)


def _export_velocity_policy(runner, path: str) -> None:
  policy_path = path.split("model")[0]
  filename = os.path.basename(os.path.dirname(policy_path)) + ".onnx"
  if runner.alg.policy.actor_obs_normalization:
    normalizer = runner.alg.policy.actor_obs_normalizer
  else:
    normalizer = None
  export_velocity_policy_as_onnx(
    runner.alg.policy,
    normalizer=normalizer,
    path=policy_path,
    filename=filename,
  )
  run_name = wandb.run.name if runner.logger_type == "wandb" and wandb.run else "local"
  attach_onnx_metadata(
    runner.env.unwrapped,
    run_name,  # type: ignore[arg-type]
    path=policy_path,
    filename=filename,
  )
  if runner.logger_type in ["wandb"]:
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
    super().save(path, infos)
    _export_velocity_policy(self, path)
