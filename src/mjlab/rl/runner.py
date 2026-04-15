from __future__ import annotations

import copy

import torch
from rsl_rl.runners import OnPolicyRunner

try:
  from rsl_rl.runners import OffPolicyRunner
except ImportError:

  class OffPolicyRunner:  # type: ignore[no-redef]
    def __init__(self, *args, **kwargs):
      raise ImportError(
        "OffPolicyRunner is not available in the installed rsl_rl package."
      )

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


class MjlabOnPolicyRunner(OnPolicyRunner):
  """Base runner that persists environment state across checkpoints."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, self._translate_train_cfg(train_cfg), log_dir, device)
    # Preserve the legacy access pattern used by mjlab2 code and tests.
    self.alg.policy = self.alg.actor  # type: ignore[attr-defined]

  def save(self, path: str, infos=None):
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = {**(infos or {}), "env_state": env_state}
    torch.save(saved_dict, path)
    if hasattr(self.logger, "writer") and self.logger.writer is not None:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self, path: str, load_optimizer: bool = True, map_location: str | None = None
  ):
    loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
    if "actor_state_dict" in loaded_dict or "critic_state_dict" in loaded_dict:
      load_cfg = {
        "actor": True,
        "critic": True,
        "optimizer": load_optimizer,
        "iteration": True,
        "rnd": True,
      }
      load_iteration = self.alg.load(loaded_dict, load_cfg, strict=True)
    else:
      legacy_dict = {
        "actor_state_dict": loaded_dict["model_state_dict"],
        "optimizer_state_dict": loaded_dict["optimizer_state_dict"],
        "iter": loaded_dict.get("iter", 0),
        "infos": loaded_dict.get("infos"),
      }
      load_cfg = {
        "actor": True,
        "critic": False,
        "optimizer": load_optimizer,
        "iteration": True,
        "rnd": True,
      }
      load_iteration = self.alg.load(legacy_dict, load_cfg, strict=True)
      loaded_dict = legacy_dict

    infos = loaded_dict.get("infos") or {}
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]
    return infos

  def _translate_train_cfg(self, train_cfg: dict) -> dict:
    cfg = copy.deepcopy(train_cfg)
    policy_cfg = dict(cfg.pop("policy", {}))
    algorithm_cfg = dict(cfg.get("algorithm", {}))
    activation = policy_cfg.get("activation", "elu")

    actor_cfg = {
      "class_name": "MLPModel",
      "hidden_dims": policy_cfg.pop("actor_hidden_dims", (128, 128, 128)),
      "activation": activation,
      "obs_normalization": policy_cfg.pop("actor_obs_normalization", False),
      "distribution_cfg": {
        "class_name": "GaussianDistribution",
        "init_std": policy_cfg.pop("init_noise_std", 1.0),
        "std_type": policy_cfg.pop("noise_std_type", "scalar"),
      },
    }
    critic_cfg = {
      "class_name": "MLPModel",
      "hidden_dims": policy_cfg.pop("critic_hidden_dims", (128, 128, 128)),
      "activation": activation,
      "obs_normalization": policy_cfg.pop("critic_obs_normalization", False),
    }

    cfg["actor"] = actor_cfg
    cfg["critic"] = critic_cfg
    cfg["algorithm"] = algorithm_cfg
    cfg["algorithm"]["class_name"] = cfg["algorithm"].get("class_name", "PPO")
    return cfg


class MjlabFastTD3Runner(OffPolicyRunner):
  """Off-policy runner adapter that keeps mjlab2 compatibility logic local."""

  env: RslRlVecEnvWrapper

  def __init__(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None = None,
    device: str = "cpu",
  ) -> None:
    super().__init__(env, self._translate_train_cfg(train_cfg), log_dir, device)
    self.alg.policy = self.alg.actor  # type: ignore[attr-defined]

  def save(self, path: str, infos=None):
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    saved_dict = self.alg.save()
    saved_dict["iter"] = self.current_learning_iteration
    saved_dict["infos"] = {**(infos or {}), "env_state": env_state}
    torch.save(saved_dict, path)
    if hasattr(self.logger, "writer") and self.logger.writer is not None:
      self.logger.save_model(path, self.current_learning_iteration)

  def load(
    self, path: str, load_optimizer: bool = True, map_location: str | None = None
  ):
    loaded_dict = torch.load(path, weights_only=False, map_location=map_location)
    load_cfg = {
      "actor": True,
      "critic1": True,
      "critic2": True,
      "critic1_target": True,
      "critic2_target": True,
      "actor_optimizer": load_optimizer,
      "critic_optimizer": load_optimizer,
      "replay_buffer": True,
      "reward_normalizer": True,
      "iteration": True,
    }
    load_iteration = self.alg.load(loaded_dict, load_cfg, strict=True)

    infos = loaded_dict.get("infos") or {}
    if infos and "env_state" in infos:
      self.env.unwrapped.common_step_counter = infos["env_state"]["common_step_counter"]
    if load_iteration:
      self.current_learning_iteration = loaded_dict["iter"]
    return infos

  def _translate_train_cfg(self, train_cfg: dict) -> dict:
    cfg = copy.deepcopy(train_cfg)
    policy_cfg = dict(cfg.pop("policy", {}))
    algorithm_cfg = dict(cfg.get("algorithm", {}))
    activation = policy_cfg.get("activation", "relu")

    actor_cfg = {
      "class_name": policy_cfg.pop("class_name", "FastTD3Actor"),
      "hidden_dims": policy_cfg.pop("actor_hidden_dims", (512, 256, 128)),
      "activation": activation,
      "obs_normalization": policy_cfg.pop("actor_obs_normalization", True),
      "init_scale": policy_cfg.pop("init_scale", 0.01),
      "std_min": policy_cfg.pop("std_min", 0.05),
      "std_max": policy_cfg.pop("std_max", 0.8),
    }
    critic_cfg = {
      "class_name": policy_cfg.pop("critic_class_name", "FastTD3Critic"),
      "hidden_dims": policy_cfg.pop("critic_hidden_dims", (1024, 512, 256)),
      "activation": activation,
      "obs_normalization": policy_cfg.pop("critic_obs_normalization", True),
    }

    cfg["actor"] = actor_cfg
    cfg["critic"] = critic_cfg
    cfg["algorithm"] = algorithm_cfg
    cfg["algorithm"]["class_name"] = cfg["algorithm"].get("class_name", "FastTD3")
    for key in (
      "num_learning_epochs",
      "num_mini_batches",
      "entropy_coef",
      "desired_kl",
      "value_loss_coef",
      "use_clipped_value_loss",
      "clip_param",
      "normalize_advantage_per_mini_batch",
      "schedule",
      "lam",
      "rnd_cfg",
      "symmetry_cfg",
      "multi_gpu_cfg",
      "policy_delay",
      "learning_rate",
    ):
      cfg["algorithm"].pop(key, None)
    return cfg
