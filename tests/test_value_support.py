"""Tests for analytic categorical value support estimation."""

from importlib import util
from pathlib import Path
import sys
import types

import pytest
import torch


def _load_module(module_name: str, path: Path):
  spec = util.spec_from_file_location(module_name, path)
  assert spec is not None and spec.loader is not None
  module = util.module_from_spec(spec)
  sys.modules[module_name] = module
  spec.loader.exec_module(module)
  return module


def test_joint_pos_limits_bound_uses_hard_minus_soft_range() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  value_support = _load_module(
    "mjlab_test_value_support_bounds",
    repo_root / "src" / "mjlab" / "rl" / "value_support.py",
  )

  asset = types.SimpleNamespace(
    data=types.SimpleNamespace(
      joint_pos_limits=torch.tensor([[[0.0, 1.0], [-2.0, 2.0]]]),
      soft_joint_pos_limits=torch.tensor([[[0.1, 0.9], [-1.5, 1.5]]]),
    )
  )
  asset_cfg = types.SimpleNamespace(name="robot", joint_ids=[0, 1])
  env = types.SimpleNamespace(scene={"robot": asset})

  bound = value_support._joint_pos_limits_bound(env, asset_cfg)

  assert bound == pytest.approx(1.2)


def test_estimate_value_support_uses_finite_horizon_geometric_sum() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  value_support = _load_module(
    "mjlab_test_value_support_finite",
    repo_root / "src" / "mjlab" / "rl" / "value_support.py",
  )

  reward_term = types.SimpleNamespace(
    func=lambda env: None,
    weight=2.0,
    params={},
  )
  reward_term.func.__name__ = "track_linear_velocity"
  env = types.SimpleNamespace(
    step_dt=0.5,
    cfg=types.SimpleNamespace(scale_rewards_by_dt=True, is_finite_horizon=True),
    max_episode_length=3,
    reward_manager=types.SimpleNamespace(cfg={"track": reward_term}),
    scene={},
  )

  estimate = value_support.estimate_value_support(env, gamma=0.5)

  assert estimate.reward_min == 0.0
  assert estimate.reward_max == 1.0
  assert estimate.factor == pytest.approx(1.75)
  assert estimate.return_min == 0.0
  assert estimate.return_max == pytest.approx(1.75)


def test_estimate_value_support_raises_on_unbounded_term() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  value_support = _load_module(
    "mjlab_test_value_support_error",
    repo_root / "src" / "mjlab" / "rl" / "value_support.py",
  )

  reward_term = types.SimpleNamespace(
    func=lambda env: None,
    weight=-0.1,
    params={},
  )
  reward_term.func.__name__ = "action_rate_l2"
  env = types.SimpleNamespace(
    step_dt=1.0,
    cfg=types.SimpleNamespace(scale_rewards_by_dt=False, is_finite_horizon=False),
    max_episode_length=10,
    reward_manager=types.SimpleNamespace(cfg={"action_rate_l2": reward_term}),
    scene={},
  )

  with pytest.raises(value_support.ValueSupportBoundError, match="action_rate_l2"):
    value_support.estimate_value_support(env, gamma=0.99)


def test_estimate_value_support_accepts_tracking_exp_rewards() -> None:
  repo_root = Path(__file__).resolve().parents[1]
  value_support = _load_module(
    "mjlab_test_value_support_tracking_exp",
    repo_root / "src" / "mjlab" / "rl" / "value_support.py",
  )

  reward_term = types.SimpleNamespace(
    func=lambda env: None,
    weight=0.5,
    params={},
  )
  reward_term.func.__name__ = "motion_global_anchor_position_error_exp"
  env = types.SimpleNamespace(
    step_dt=1.0,
    cfg=types.SimpleNamespace(scale_rewards_by_dt=False, is_finite_horizon=True),
    max_episode_length=2,
    reward_manager=types.SimpleNamespace(cfg={"tracking": reward_term}),
    scene={},
  )

  estimate = value_support.estimate_value_support(env, gamma=0.5)

  assert estimate.reward_min == 0.0
  assert estimate.reward_max == 0.5
  assert estimate.return_max == pytest.approx(0.75)
