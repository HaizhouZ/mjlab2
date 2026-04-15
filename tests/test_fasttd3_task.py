"""Regression tests for the Yam FastTD3 config and adapter."""

from dataclasses import asdict
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


def test_yam_fasttd3_config_matches_reference_support_settings() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  rl_config = _load_module(
    "mjlab_test_rl_config_fasttd3",
    repo_root / "src" / "mjlab" / "rl" / "config.py",
  )

  fake_rl = types.ModuleType("mjlab.rl")
  fake_rl.RslRlFastTd3RunnerCfg = rl_config.RslRlFastTd3RunnerCfg
  fake_rl.RslRlReppoRunnerCfg = rl_config.RslRlReppoRunnerCfg
  sys.modules["mjlab.rl"] = fake_rl

  yam_rl_cfg = _load_module(
    "mjlab_test_fasttd3_yam_rl_cfg",
    repo_root / "src" / "mjlab" / "tasks" / "manipulation" / "config" / "yam" / "rl_cfg.py",
  )

  cfg = yam_rl_cfg.yam_lift_cube_fasttd3_runner_cfg()

  assert cfg.class_name == "FastTD3Runner"
  assert cfg.policy.class_name == "FastTD3Actor"
  assert cfg.policy.critic_class_name == "FastTD3Critic"
  assert cfg.policy.activation == "relu"
  assert cfg.policy.actor_obs_normalization is True
  assert cfg.policy.critic_obs_normalization is True
  assert cfg.policy.init_scale == 0.01
  assert cfg.policy.std_min == 0.05
  assert cfg.policy.std_max == 0.8
  assert cfg.algorithm.class_name == "FastTD3"
  assert cfg.algorithm.replay_size == 50_000
  assert cfg.algorithm.batch_size == 128
  assert cfg.algorithm.learning_starts == 1_000
  assert cfg.algorithm.policy_frequency == 2
  assert cfg.algorithm.actor_learning_rate == 3.0e-4
  assert cfg.algorithm.critic_learning_rate == 3.0e-4
  assert cfg.algorithm.weight_decay == 0.1
  assert cfg.algorithm.num_atoms == 101
  assert cfg.algorithm.v_min == -50.0
  assert cfg.algorithm.v_max == 50.0
  assert cfg.algorithm.use_cdq is True


def test_fasttd3_adapter_strips_ppo_only_fields(monkeypatch) -> None:
  repo_root = Path(__file__).resolve().parents[1]
  rl_config = _load_module(
    "mjlab_test_rl_config_fasttd3_adapter",
    repo_root / "src" / "mjlab" / "rl" / "config.py",
  )

  fake_mjlab = types.ModuleType("mjlab")
  fake_mjlab.__path__ = []  # type: ignore[attr-defined]
  fake_mjlab_rl = types.ModuleType("mjlab.rl")
  fake_mjlab_rl.__path__ = []  # type: ignore[attr-defined]
  fake_vecenv = types.ModuleType("mjlab.rl.vecenv_wrapper")
  fake_vecenv.RslRlVecEnvWrapper = object
  fake_rsl_rl = types.ModuleType("rsl_rl")
  fake_rsl_rl.__path__ = []  # type: ignore[attr-defined]
  fake_rsl_rl_runners = types.ModuleType("rsl_rl.runners")

  class _OffPolicyRunner:
    pass

  class _OnPolicyRunner:
    pass

  fake_rsl_rl_runners.OffPolicyRunner = _OffPolicyRunner
  fake_rsl_rl_runners.OnPolicyRunner = _OnPolicyRunner

  monkeypatch.setitem(sys.modules, "mjlab", fake_mjlab)
  monkeypatch.setitem(sys.modules, "mjlab.rl", fake_mjlab_rl)
  monkeypatch.setitem(sys.modules, "mjlab.rl.vecenv_wrapper", fake_vecenv)
  monkeypatch.setitem(sys.modules, "rsl_rl", fake_rsl_rl)
  monkeypatch.setitem(sys.modules, "rsl_rl.runners", fake_rsl_rl_runners)

  runner_mod = _load_module(
    "mjlab_test_fasttd3_runner",
    repo_root / "src" / "mjlab" / "rl" / "runner.py",
  )

  cfg = rl_config.RslRlFastTd3RunnerCfg(
    experiment_name="yam_lift_cube_fasttd3",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )
  cfg_dict = asdict(cfg)
  translated = runner_mod.MjlabFastTD3Runner._translate_train_cfg(None, cfg_dict)

  assert translated["actor"]["class_name"] == "FastTD3Actor"
  assert translated["critic"]["class_name"] == "FastTD3Critic"
  assert translated["algorithm"]["class_name"] == "FastTD3"
  assert translated["actor"]["hidden_dims"] == (512, 256, 128)
  assert translated["critic"]["hidden_dims"] == (1024, 512, 256)
  assert translated["algorithm"]["replay_size"] == 100_000
  assert translated["algorithm"]["batch_size"] == 256
  assert translated["algorithm"]["learning_starts"] == 1_000
  assert translated["algorithm"]["policy_frequency"] == 2
  assert translated["algorithm"]["actor_learning_rate"] == 3.0e-4
  assert translated["algorithm"]["actor_learning_rate_end"] == 3.0e-4
  assert translated["algorithm"]["critic_learning_rate"] == 3.0e-4
  assert translated["algorithm"]["critic_learning_rate_end"] == 3.0e-4
  assert translated["algorithm"]["weight_decay"] == 0.1
  assert translated["algorithm"]["n_steps"] == 1
  assert translated["algorithm"]["num_atoms"] == 101
  assert translated["algorithm"]["v_min"] == -250.0
  assert translated["algorithm"]["v_max"] == 250.0
  assert translated["algorithm"]["use_cdq"] is True
  assert translated["actor"]["activation"] == "relu"
  assert translated["actor"]["init_scale"] == 0.01
  assert translated["actor"]["std_min"] == 0.05
  assert translated["actor"]["std_max"] == 0.8
  assert "num_learning_epochs" not in translated["algorithm"]
  assert "num_mini_batches" not in translated["algorithm"]
  assert "entropy_coef" not in translated["algorithm"]
  assert "desired_kl" not in translated["algorithm"]
  assert "value_loss_coef" not in translated["algorithm"]
  assert "use_clipped_value_loss" not in translated["algorithm"]
  assert "clip_param" not in translated["algorithm"]
  assert "normalize_advantage_per_mini_batch" not in translated["algorithm"]
  assert "schedule" not in translated["algorithm"]
  assert "lam" not in translated["algorithm"]
  assert "rnd_cfg" not in translated["algorithm"]
  assert "symmetry_cfg" not in translated["algorithm"]
  assert "multi_gpu_cfg" not in translated["algorithm"]
  assert "policy_delay" not in translated["algorithm"]
  assert "learning_rate" not in translated["algorithm"]
