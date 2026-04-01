"""Regression tests for the Yam task REPPO wiring."""

from mjlab.tasks.registry import load_rl_cfg, load_runner_cls
from rsl_rl.runners import ReppoRunner


def test_yam_task_uses_reppo_runner() -> None:
  """The Yam task should opt into the REPPO runner."""
  runner_cls = load_runner_cls("Mjlab-Lift-Cube-Yam")
  assert runner_cls is ReppoRunner


def test_yam_task_uses_reppo_policy_config() -> None:
  """The Yam task RL config should request REPPO policy modules."""
  rl_cfg = load_rl_cfg("Mjlab-Lift-Cube-Yam")
  assert rl_cfg.class_name == "ReppoRunner"
  assert rl_cfg.policy.class_name == "ReppoPolicy"
  assert rl_cfg.policy.kl_start == 0.01
  assert rl_cfg.algorithm.class_name == "Reppo"
  assert rl_cfg.algorithm.num_atoms == 151
  assert rl_cfg.algorithm.kl_bound == 0.1
