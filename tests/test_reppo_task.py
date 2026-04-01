"""Regression tests for the Yam task REPPO config."""

from importlib import util
from pathlib import Path
import sys
import types


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
  rl_config = _load_module("mjlab_test_rl_config", repo_root / "src" / "mjlab" / "rl" / "config.py")

  fake_rl = types.ModuleType("mjlab.rl")
  fake_rl.RslRlReppoRunnerCfg = rl_config.RslRlReppoRunnerCfg
  fake_rl.RslRlFastTd3RunnerCfg = rl_config.RslRlFastTd3RunnerCfg
  sys.modules["mjlab.rl"] = fake_rl

  yam_rl_cfg = _load_module(
    "mjlab_test_yam_rl_cfg",
    repo_root / "src" / "mjlab" / "tasks" / "manipulation" / "config" / "yam" / "rl_cfg.py",
  )

  cfg = yam_rl_cfg.yam_lift_cube_reppo_runner_cfg()

  assert cfg.class_name == "ReppoRunner"
  assert cfg.experiment_name == "yam_lift_cube"
  assert cfg.policy.class_name == "ReppoPolicy"
  assert cfg.policy.critic_class_name == "ReppoCritic"
  assert cfg.policy.actor_obs_normalization is True
  assert cfg.policy.critic_obs_normalization is True
  assert cfg.policy.actor_hidden_dims == (512, 256, 128)
  assert cfg.policy.critic_hidden_dims == (512, 256, 128)
  assert cfg.policy.ent_start == 0.001
  assert cfg.policy.kl_start == 0.01
  assert cfg.algorithm.class_name == "Reppo"
  assert cfg.algorithm.num_learning_epochs == 4
  assert cfg.algorithm.num_mini_batches == 32
  assert cfg.algorithm.learning_rate == 3.0e-4
  assert cfg.algorithm.num_atoms == 151
  assert cfg.algorithm.vmin == 0.0
  assert cfg.algorithm.vmax == 150.0
  assert cfg.algorithm.kl_bound == 0.1
  assert cfg.algorithm.num_action_samples == 64
