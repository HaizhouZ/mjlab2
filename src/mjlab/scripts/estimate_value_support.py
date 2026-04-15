"""Estimate categorical value support for a task from analytic reward bounds."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import sys

from prettytable import PrettyTable

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl.value_support import estimate_value_support
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiMotionCommandCfg
from mjlab.utils import parse_choice_and_dataclass


@dataclass(frozen=True)
class EstimateValueSupportCfg:
  device: str = "cpu"
  gamma: float | None = None
  motion_file: str | None = None
  motion_dir: str | None = None


def _configure_tracking_motion_source(env_cfg, cfg: EstimateValueSupportCfg) -> None:
  if env_cfg.commands is None or "motion" not in env_cfg.commands:
    return

  motion_cmd = env_cfg.commands["motion"]
  if isinstance(motion_cmd, MotionCommandCfg):
    if cfg.motion_file is None:
      raise ValueError(
        "Tracking tasks with a single motion command require `motion_file`."
      )
    motion_file = Path(cfg.motion_file)
    if not motion_file.exists():
      raise FileNotFoundError(f"Motion file not found: {motion_file}")
    print(f"[INFO] Using local motion file: {motion_file}")
    motion_cmd.motion_file = str(motion_file)
    return

  if isinstance(motion_cmd, MultiMotionCommandCfg):
    if cfg.motion_dir is None:
      raise ValueError(
        "Tracking tasks with a multi-motion command require `motion_dir`."
      )
    motion_dir = Path(cfg.motion_dir)
    if not motion_dir.exists():
      raise FileNotFoundError(f"Motion directory not found: {motion_dir}")
    print(f"[INFO] Using local motion directory: {motion_dir}")
    motion_cmd.motion_dir = str(motion_dir)
    motion_cmd.motion_name_pattern = [".*"]


def run_estimate(task_id: str, cfg: EstimateValueSupportCfg) -> None:
  env_cfg = load_env_cfg(task_id)
  rl_cfg = load_rl_cfg(task_id)
  gamma = cfg.gamma if cfg.gamma is not None else float(rl_cfg.algorithm.gamma)
  _configure_tracking_motion_source(env_cfg, cfg)

  env = ManagerBasedRlEnv(env_cfg, device=cfg.device)
  try:
    estimate = estimate_value_support(env, gamma=gamma)
  finally:
    env.close()

  print(f"Task: {task_id}")
  print(f"Gamma: {gamma}")
  if estimate.horizon is None:
    print(f"Horizon: infinite (factor={estimate.factor:.6f})")
  else:
    print(f"Horizon: {estimate.horizon} steps (factor={estimate.factor:.6f})")
  print(
    "Step reward bound: "
    f"[{estimate.reward_min:.6f}, {estimate.reward_max:.6f}]"
  )
  print(
    "Return bound: "
    f"[{estimate.return_min:.6f}, {estimate.return_max:.6f}]"
  )

  algorithm_name = type(rl_cfg).__name__
  if "FastTd3" in algorithm_name:
    print("Suggested overrides: agent.algorithm.v_min / agent.algorithm.v_max")
  elif "Reppo" in algorithm_name:
    print("Suggested overrides: agent.algorithm.vmin / agent.algorithm.vmax")

  table = PrettyTable(
    ["Term", "Callable", "Weight", "Raw Min", "Raw Max", "Step Min", "Step Max"]
  )
  table.align["Term"] = "l"
  table.align["Callable"] = "l"
  for term in estimate.term_bounds:
    table.add_row(
      [
        term.name,
        term.func_name,
        f"{term.weight:.6f}",
        f"{term.raw_min:.6f}",
        f"{term.raw_max:.6f}",
        f"{term.weighted_min:.6f}",
        f"{term.weighted_max:.6f}",
      ]
    )
  print(table)


def main() -> None:
  import mjlab.tasks  # noqa: F401

  task_id, cfg = parse_choice_and_dataclass(
    list_tasks(),
    EstimateValueSupportCfg,
    prog=sys.argv[0],
    description=__doc__,
  )
  run_estimate(task_id, cfg)


if __name__ == "__main__":
  main()
