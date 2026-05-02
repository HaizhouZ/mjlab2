"""Regression tests for the Yam task REPPO config."""

import sys
import types
from dataclasses import asdict
from importlib import util
from pathlib import Path

import yaml


def _load_module(module_name: str, path: Path):
  spec = util.spec_from_file_location(module_name, path)
  assert spec is not None and spec.loader is not None
  module = util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


def test_yam_reppo_config_matches_mjplayground_settings() -> None:
  """The Yam REPPO config should mirror the MuJoCo playground example."""
  repo_root = Path(__file__).resolve().parents[1]
  rl_config = _load_module(
    "mjlab_test_rl_config", repo_root / "src" / "mjlab" / "rl" / "config.py"
  )

  fake_rl = types.ModuleType("mjlab.rl")
  fake_rl.RslRlReppoRunnerCfg = rl_config.RslRlReppoRunnerCfg
  fake_rl.RslRlFastTd3RunnerCfg = rl_config.RslRlFastTd3RunnerCfg
  sys.modules["mjlab.rl"] = fake_rl

  yam_rl_cfg = _load_module(
    "mjlab_test_yam_rl_cfg",
    repo_root
    / "src"
    / "mjlab"
    / "tasks"
    / "manipulation"
    / "config"
    / "yam"
    / "rl_cfg.py",
  )

  cfg = yam_rl_cfg.yam_lift_cube_reppo_runner_cfg()

  assert cfg.class_name == "OnPolicyRunner"
  assert cfg.experiment_name == "yam_lift_cube"
  assert cfg.policy.class_name == "ActorQ"
  assert cfg.policy.actor_obs_normalization is True
  assert cfg.policy.critic_obs_normalization is True
  assert cfg.policy.actor_hidden_dims == (512, 256, 128)
  assert cfg.policy.critic_hidden_dims == (512, 256, 128)
  assert cfg.policy.init_alpha_temp == 0.001
  assert cfg.policy.init_alpha_kl == 0.01
  assert cfg.policy.num_critic_bins == 151
  assert cfg.policy.vmin == 0.0
  assert cfg.policy.vmax == 150.0
  assert cfg.algorithm.class_name == "REPPO"
  assert cfg.algorithm.num_learning_epochs == 4
  assert cfg.algorithm.num_mini_batches == 32
  assert cfg.algorithm.learning_rate == 3.0e-4
  assert cfg.algorithm.desired_kl == 0.1
  assert cfg.algorithm.target_entropy == -0.5
  assert cfg.algorithm.actor_route == "reppo"
  assert cfg.algorithm.clip_param == 0.2
  assert cfg.algorithm.ppo_entropy_coef == 0.01
  assert cfg.algorithm.ppo_value_loss_coef == 1.0
  assert cfg.algorithm.ppo_schedule == "adaptive"


def test_g1_velocity_reppo_config_uses_estimated_value_support() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  rl_config = _load_module(
    "mjlab_test_rl_config_velocity_reppo",
    repo_root / "src" / "mjlab" / "rl" / "config.py",
  )

  fake_rl = types.ModuleType("mjlab.rl")
  fake_rl.RslRlOnPolicyRunnerCfg = rl_config.RslRlOnPolicyRunnerCfg
  fake_rl.RslRlPpoActorCriticCfg = rl_config.RslRlPpoActorCriticCfg
  fake_rl.RslRlPpoAlgorithmCfg = rl_config.RslRlPpoAlgorithmCfg
  fake_rl.RslRlReppoActorCriticCfg = rl_config.RslRlReppoActorCriticCfg
  fake_rl.RslRlReppoAlgorithmCfg = rl_config.RslRlReppoAlgorithmCfg
  fake_rl.RslRlReppoRunnerCfg = rl_config.RslRlReppoRunnerCfg
  sys.modules["mjlab.rl"] = fake_rl

  velocity_rl_cfg = _load_module(
    "mjlab_test_velocity_g1_rl_cfg",
    repo_root / "src" / "mjlab" / "tasks" / "velocity" / "config" / "g1" / "rl_cfg.py",
  )

  cfg = velocity_rl_cfg.unitree_g1_reppo_runner_cfg()

  assert cfg.class_name == "OnPolicyRunner"
  assert cfg.experiment_name == "g1_velocity_reppo"
  assert cfg.policy.class_name == "ActorQ"
  assert cfg.algorithm.class_name == "REPPO"
  assert cfg.policy.num_critic_bins == 151
  assert cfg.policy.vmin == -20.0
  assert cfg.policy.vmax == 15.0
  assert cfg.algorithm.gamma == 0.99


def test_velocity_reppo_yaml_matches_runtime_config() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  velocity_rl_cfg = _load_module(
    "mjlab_test_velocity_g1_rl_cfg_yaml",
    repo_root / "src" / "mjlab" / "tasks" / "velocity" / "config" / "g1" / "rl_cfg.py",
  )

  cfg = velocity_rl_cfg.unitree_g1_reppo_runner_cfg()
  yaml_path = repo_root / "conf" / "g1_velocity_reppo_runtime.yaml"

  with open(yaml_path, "r", encoding="utf-8") as f:
    overrides = yaml.safe_load(f)

  from mjlab.utils.config_loader import apply_config_overrides

  apply_config_overrides(cfg, overrides["agent"], recursive=True)
  cfg_dict = asdict(cfg)

  assert cfg_dict["policy"]["class_name"] == "ActorQ"
  assert cfg_dict["algorithm"]["class_name"] == "REPPO"
  assert cfg_dict["policy"]["vmin"] == -20.0
  assert cfg_dict["policy"]["vmax"] == 15.0
