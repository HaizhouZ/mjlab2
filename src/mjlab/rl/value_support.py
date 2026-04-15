from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
  from mjlab.envs import ManagerBasedRlEnv


class ValueSupportBoundError(ValueError):
  """Raised when a reward term cannot be bounded analytically."""


@dataclass(frozen=True)
class RewardTermBound:
  name: str
  raw_min: float
  raw_max: float
  weighted_min: float
  weighted_max: float
  weight: float
  func_name: str


@dataclass(frozen=True)
class ValueSupportEstimate:
  reward_min: float
  reward_max: float
  return_min: float
  return_max: float
  horizon: int | None
  factor: float
  term_bounds: tuple[RewardTermBound, ...]


def estimate_value_support(
  env: "ManagerBasedRlEnv",
  *,
  gamma: float,
) -> ValueSupportEstimate:
  """Estimate categorical value support from strict analytic reward bounds."""
  term_bounds = tuple(_compute_term_bounds(env))
  reward_min = sum(term.weighted_min for term in term_bounds)
  reward_max = sum(term.weighted_max for term in term_bounds)
  factor, horizon = _discount_factor(env, gamma)
  return ValueSupportEstimate(
    reward_min=reward_min,
    reward_max=reward_max,
    return_min=reward_min * factor,
    return_max=reward_max * factor,
    horizon=horizon,
    factor=factor,
    term_bounds=term_bounds,
  )


def _compute_term_bounds(env: "ManagerBasedRlEnv") -> list[RewardTermBound]:
  bounds = []
  scale = env.step_dt if env.cfg.scale_rewards_by_dt else 1.0
  for name, term_cfg in env.reward_manager.cfg.items():
    if term_cfg.weight == 0.0:
      bounds.append(
        RewardTermBound(
          name=name,
          raw_min=0.0,
          raw_max=0.0,
          weighted_min=0.0,
          weighted_max=0.0,
          weight=0.0,
          func_name=_callable_name(term_cfg.func),
        )
      )
      continue
    raw_min, raw_max = _term_raw_bounds(env, name, term_cfg)
    weighted_min, weighted_max = _apply_weight(raw_min, raw_max, term_cfg.weight * scale)
    bounds.append(
      RewardTermBound(
        name=name,
        raw_min=raw_min,
        raw_max=raw_max,
        weighted_min=weighted_min,
        weighted_max=weighted_max,
        weight=term_cfg.weight,
        func_name=_callable_name(term_cfg.func),
      )
    )
  return bounds


def _term_raw_bounds(
  env: "ManagerBasedRlEnv",
  term_name: str,
  term_cfg: Any,
) -> tuple[float, float]:
  func = term_cfg.func
  func_name = _callable_name(func)

  bounded_unit_interval = {
    "is_alive",
    "is_terminated",
    "bring_object_reward",
    "track_linear_velocity",
    "track_angular_velocity",
    "flat_orientation",
    "flat_orientation_l2",
    "motion_global_anchor_position_error_exp",
    "motion_global_anchor_orientation_error_exp",
    "motion_relative_body_position_error_exp",
    "motion_relative_body_orientation_error_exp",
    "motion_global_body_linear_velocity_error_exp",
    "motion_global_body_angular_velocity_error_exp",
    "posture",
    "variable_posture",
  }
  if func_name in bounded_unit_interval:
    return 0.0, 1.0

  if func_name == "staged_position_reward":
    return 0.0, 2.0

  if func_name == "feet_air_time":
    sensor = env.scene[term_cfg.params["sensor_name"]]
    current_air_time = sensor.data.current_air_time
    if current_air_time is None:
      raise ValueSupportBoundError(
        f"Reward term '{term_name}' uses feet_air_time but sensor air-time data is unavailable."
      )
    return 0.0, float(current_air_time.shape[1])

  if func_name == "joint_pos_limits":
    return 0.0, _joint_pos_limits_bound(env, term_cfg.params["asset_cfg"])

  raise ValueSupportBoundError(
    f"Reward term '{term_name}' with callable '{func_name}' has no analytic bound implementation."
  )


def _joint_pos_limits_bound(env: "ManagerBasedRlEnv", asset_cfg: Any) -> float:
  asset = env.scene[asset_cfg.name]
  joint_ids = asset_cfg.joint_ids
  hard_limits = asset.data.joint_pos_limits[0, joint_ids]
  soft_limits = asset.data.soft_joint_pos_limits[0, joint_ids]
  lower_excess = (soft_limits[:, 0] - hard_limits[:, 0]).clamp_min(0.0)
  upper_excess = (hard_limits[:, 1] - soft_limits[:, 1]).clamp_min(0.0)
  return float((lower_excess + upper_excess).sum().item())


def _apply_weight(raw_min: float, raw_max: float, weight: float) -> tuple[float, float]:
  if weight >= 0.0:
    return raw_min * weight, raw_max * weight
  return raw_max * weight, raw_min * weight


def _discount_factor(env: "ManagerBasedRlEnv", gamma: float) -> tuple[float, int | None]:
  if gamma < 0.0:
    raise ValueSupportBoundError(f"Expected gamma >= 0, got {gamma}.")

  if env.cfg.is_finite_horizon:
    horizon = env.max_episode_length
    if gamma == 1.0:
      return float(horizon), horizon
    return float((1.0 - gamma**horizon) / (1.0 - gamma)), horizon

  if gamma >= 1.0:
    raise ValueSupportBoundError(
      "Infinite-horizon tasks require gamma < 1.0 for a finite analytic return bound."
    )
  return float(1.0 / (1.0 - gamma)), None


def _callable_name(func: Any) -> str:
  if hasattr(func, "__name__"):
    return str(func.__name__)
  return type(func).__name__
