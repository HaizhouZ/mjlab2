from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import rsl_rl
import torch
from rsl_rl.algorithms.reppo import REPPO
from rsl_rl.extensions import resolve_rnd_config, resolve_symmetry_config
from rsl_rl.storage.reppo_rollout_storage import ReppoRolloutStorage
from rsl_rl.utils import resolve_callable, resolve_obs_groups
from rsl_rl.utils.logger import Logger

from mjlab.rl.vecenv_wrapper import RslRlVecEnvWrapper


def _load_runner_class(module_name: str, class_name: str):
  module_path = (
    Path(rsl_rl.__file__)
    .resolve()
    .parent.joinpath(*module_name.split(".")[1:])
    .with_suffix(".py")
  )
  if not module_path.exists():
    raise ImportError(f"Could not locate runner module at {module_path}.")
  module_spec = importlib.util.spec_from_file_location(
    f"mjlab._compat_{module_name.rsplit('.', 1)[-1]}",
    module_path,
  )
  if module_spec is None or module_spec.loader is None:
    raise ImportError(f"Could not load module spec for {module_name!r}.")
  module = importlib.util.module_from_spec(module_spec)
  module_spec.loader.exec_module(module)
  return getattr(module, class_name)


OnPolicyRunner = _load_runner_class("rsl_rl.runners.on_policy_runner", "OnPolicyRunner")

try:
  OffPolicyRunner = _load_runner_class(
    "rsl_rl.runners.off_policy_runner", "OffPolicyRunner"
  )
except ImportError:

  class OffPolicyRunner:  # type: ignore[no-redef]
    def __init__(self, *args, **kwargs):
      raise ImportError(
        "OffPolicyRunner is not available in the installed rsl_rl package."
      )


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
    if self._uses_reppo(train_cfg):
      self._init_reppo(env, train_cfg, log_dir, device)
    else:
      super().__init__(env, self._translate_train_cfg(train_cfg), log_dir, device)
      # Preserve the legacy access pattern used by mjlab2 code and tests.
      self.alg.policy = self.alg.actor  # type: ignore[attr-defined]

  def save(self, path: str, infos=None):
    env_state = {"common_step_counter": self.env.unwrapped.common_step_counter}
    if isinstance(self.alg, REPPO):
      saved_dict = {
        "policy_state_dict": self.alg.policy.state_dict(),
        "optimizer_state_dict": self.alg.optimizer.state_dict(),
        "iter": self.current_learning_iteration,
        "infos": {**(infos or {}), "env_state": env_state},
      }
      if self.alg.rnd:
        saved_dict["rnd_state_dict"] = self.alg.rnd.state_dict()
        saved_dict["rnd_optimizer_state_dict"] = self.alg.rnd_optimizer.state_dict()
      torch.save(saved_dict, path)
      if hasattr(self.logger, "writer") and self.logger.writer is not None:
        self.logger.save_model(path, self.current_learning_iteration)
      return

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
    if isinstance(self.alg, REPPO):
      self.alg.policy.load_state_dict(loaded_dict["policy_state_dict"], strict=True)
      if load_optimizer and "optimizer_state_dict" in loaded_dict:
        self.alg.optimizer.load_state_dict(loaded_dict["optimizer_state_dict"])
      if self.alg.rnd and "rnd_state_dict" in loaded_dict:
        self.alg.rnd.load_state_dict(loaded_dict["rnd_state_dict"], strict=True)
        if load_optimizer and "rnd_optimizer_state_dict" in loaded_dict:
          self.alg.rnd_optimizer.load_state_dict(
            loaded_dict["rnd_optimizer_state_dict"]
          )
      infos = loaded_dict.get("infos") or {}
      if infos and "env_state" in infos:
        self.env.unwrapped.common_step_counter = infos["env_state"][
          "common_step_counter"
        ]
      self.current_learning_iteration = loaded_dict.get("iter", 0)
      return infos

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

  def get_inference_policy(self, device: str | None = None):
    if isinstance(self.alg, REPPO):
      self.alg.policy.eval()
      return self.alg.policy.to(device)
    return super().get_inference_policy(device)

  @staticmethod
  def _uses_reppo(train_cfg: dict) -> bool:
    algorithm_cfg = train_cfg.get("algorithm", {})
    return algorithm_cfg.get("class_name") == "REPPO"

  def _init_reppo(
    self,
    env: RslRlVecEnvWrapper,
    train_cfg: dict,
    log_dir: str | None,
    device: str,
  ) -> None:
    self.env = env
    self.cfg = copy.deepcopy(train_cfg)
    self.device = device

    self._configure_multi_gpu()

    obs = self.env.get_observations()
    self.alg = self._construct_reppo_algorithm(obs, self.env, self.cfg, self.device)
    self.alg.train_mode = self._reppo_train_mode  # type: ignore[attr-defined]
    self.alg.eval_mode = self._reppo_eval_mode  # type: ignore[attr-defined]
    self.alg.get_policy = self._reppo_get_policy  # type: ignore[attr-defined]
    self.logger = Logger(
      log_dir=log_dir,
      cfg=self.cfg,
      env_cfg=self.env.cfg,
      num_envs=self.env.num_envs,
      is_distributed=self.is_distributed,
      gpu_world_size=self.gpu_world_size,
      gpu_global_rank=self.gpu_global_rank,
      device=self.device,
    )
    self.current_learning_iteration = 0

  def _construct_reppo_algorithm(self, obs, env, cfg: dict, device: str) -> REPPO:
    alg_class: type[REPPO] = resolve_callable(cfg["algorithm"].pop("class_name"))  # type: ignore[assignment]
    policy_class = resolve_callable(cfg["policy"].pop("class_name"))

    default_sets = ["policy", "critic"]
    if cfg["algorithm"].get("rnd_cfg") is not None:
      default_sets.append("rnd_state")
    cfg["obs_groups"] = resolve_obs_groups(obs, cfg["obs_groups"], default_sets)
    cfg["algorithm"] = resolve_rnd_config(cfg["algorithm"], obs, cfg["obs_groups"], env)
    cfg["algorithm"] = resolve_symmetry_config(cfg["algorithm"], env)

    policy = policy_class(obs, cfg["obs_groups"], env.num_actions, **cfg["policy"]).to(
      device
    )
    storage = ReppoRolloutStorage(
      "rl",
      env.num_envs,
      cfg["num_steps_per_env"],
      obs,
      [env.num_actions],
      device,
    )
    return alg_class(
      policy,
      storage,
      device=device,
      **cfg["algorithm"],
      multi_gpu_cfg=cfg["multi_gpu"],
    )

  def _reppo_train_mode(self) -> None:
    self.alg.policy.train()
    if self.alg.rnd:
      self.alg.rnd.train()

  def _reppo_eval_mode(self) -> None:
    self.alg.policy.eval()
    if self.alg.rnd:
      self.alg.rnd.eval()

  def _reppo_get_policy(self):
    return self.alg.policy

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
