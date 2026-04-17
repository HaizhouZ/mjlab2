import os
import time
from types import SimpleNamespace

import torch
import wandb
from rsl_rl.env.vec_env import VecEnv
from rsl_rl.runners import ReppoRunner
from rsl_rl.runners.reppo_runner import check_nan

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
  run_name = (
    wandb.run.name
    if logger_type == "wandb" and wandb.run
    else "local"
  )
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


class MotionTrackingReppoRunner(ReppoRunner):
  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: VecEnv,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
    registry_name: str | None = None,
  ):
    algorithm_cfg = train_cfg.setdefault("algorithm", {})
    algorithm_cfg.setdefault("rnd_cfg", None)
    algorithm_cfg.setdefault("symmetry_cfg", None)
    super().__init__(env, train_cfg, log_dir, device)
    self.registry_name = registry_name

  def learn(self, num_learning_iterations: int, init_at_random_ep_len: bool = False) -> None:
    if init_at_random_ep_len:
      self.env.episode_length_buf = torch.randint_like(
        self.env.episode_length_buf, high=int(self.env.max_episode_length)
      )

    obs = self.env.get_observations().to(self.device)
    self.train_mode()

    if self.is_distributed:
      self.broadcast_parameters()

    self.logger.init_logging_writer()

    start_it = self.current_learning_iteration
    total_it = start_it + num_learning_iterations
    for it in range(start_it, total_it):
      start = time.time()
      with torch.inference_mode():
        for _ in range(self.cfg["num_steps_per_env"]):
          actions = self.act(obs)
          obs, rewards, dones, extras = self.env.step(actions.to(self.env.device))
          if self.cfg.get("check_for_nan", True):
            check_nan(obs, rewards, dones)
          obs = obs.to(self.device)
          rewards = rewards.to(self.device)
          dones = dones.to(self.device)
          self.process_env_step(obs, rewards, dones, extras)
          self.logger.process_env_step(rewards.detach(), dones, extras)

        stop = time.time()
        collect_time = stop - start
        start = stop

        self.compute_returns(obs)

      loss_dict, metric_dict = self.update()

      stop = time.time()
      learn_time = stop - start
      self.current_learning_iteration = it

      self.logger.log(
        it=it,
        start_it=start_it,
        total_it=total_it,
        collect_time=collect_time,
        learn_time=learn_time,
        loss_dict=loss_dict,
        metric_dict=metric_dict,
        learning_rate=self.actor_optimizer.param_groups[0]["lr"],
        action_std=self.policy.output_std,
        rnd_weight=None,
      )

      if self.gpu_global_rank == 0 and self.logger.log_dir is not None and it % self.cfg["save_interval"] == 0:
        self.save(os.path.join(self.logger.log_dir, f"model_{it}.pt"))

    if self.gpu_global_rank == 0 and self.logger.log_dir is not None:
      self.save(os.path.join(self.logger.log_dir, f"model_{self.current_learning_iteration}.pt"))
    self.logger.stop_logging_writer()

  def save(self, path: str, infos=None):
    super().save(path, infos)
    _save_tracking_policy(self, path)
