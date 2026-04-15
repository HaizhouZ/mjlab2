"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Literal, Optional, cast

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import MjlabOnPolicyRunner, RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiMotionCommandCfg
from mjlab.utils import parse_choice_and_dataclass
from mjlab.utils.gpu import select_gpus
from mjlab.utils.log_dir import get_log_dir
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import (
  add_wandb_tags,
  get_wandb_entity_and_project,
  get_wandb_motion_cache_dir,
)
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  motion_file: str | None = None
  motion_dir: str | None = None
  motion_name_pattern: list[str] = field(default_factory=lambda: [".*"])
  wandb_entity_project: str | None = None
  device: str = "cuda:0"
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  config_path: Optional[str] = None
  """Optional path to YAML config file to load defaults from."""
  config_source: Optional[str] = None
  """Optional config source: local path, http(s) URL, or wandb:// URI."""

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def _resolve_training_runtime(cfg: TrainConfig) -> tuple[str, int, int]:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    return "cpu", cfg.agent.seed, 0

  local_rank = int(os.environ.get("LOCAL_RANK", "0"))
  rank = int(os.environ.get("RANK", "0"))
  os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
  return f"cuda:{local_rank}", cfg.agent.seed + local_rank, rank


def _get_tracking_motion_cmd(
  env_cfg: ManagerBasedRlEnvCfg,
) -> MotionCommandCfg | MultiMotionCommandCfg | None:
  if env_cfg.commands is None or "motion" not in env_cfg.commands:
    return None

  motion_cmd = env_cfg.commands["motion"]
  if isinstance(motion_cmd, (MotionCommandCfg, MultiMotionCommandCfg)):
    return motion_cmd
  return None


def _configure_tracking_motion_source(
  cfg: TrainConfig,
  motion_cmd: MotionCommandCfg | MultiMotionCommandCfg,
) -> str | None:
  registry_name: str | None = None
  wandb_entity, wandb_project = get_wandb_entity_and_project(cfg.wandb_entity_project)

  if isinstance(motion_cmd, MultiMotionCommandCfg) and wandb_entity and wandb_project:
    print(f"[INFO] Loading motions from W&B: {wandb_entity}/{wandb_project}")
    motions_dir = get_wandb_motion_cache_dir(
      wandb_entity=wandb_entity, wandb_project=wandb_project
    )
    motion_cmd.motion_dir = str(motions_dir)
    motion_cmd.motion_name_pattern = cfg.motion_name_pattern
    print(f"[INFO] Using motions from: {motions_dir}")
    return None

  if cfg.registry_name:
    registry_name = cast(str, cfg.registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    import wandb

    api = wandb.Api()
    artifact = api.artifact(registry_name)
    if isinstance(motion_cmd, MotionCommandCfg):
      motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
    else:
      motion_cmd.motion_dir = str(artifact.download())
      motion_cmd.motion_name_pattern = cfg.motion_name_pattern
    return registry_name

  if cfg.motion_file is not None:
    print(f"[INFO] Using local motion file: {cfg.motion_file}")
    if isinstance(motion_cmd, MotionCommandCfg):
      motion_cmd.motion_file = cfg.motion_file
    else:
      motion_dir = cfg.motion_dir
      if motion_dir is None:
        raise ValueError("Must provide --motion-dir for multi-motion tracking tasks.")
      motion_cmd.motion_dir = motion_dir
      motion_cmd.motion_name_pattern = cfg.motion_name_pattern
    return None

  if cfg.motion_dir is not None:
    print(f"[INFO] Using local motion directory: {cfg.motion_dir}")
    if isinstance(motion_cmd, MotionCommandCfg):
      raise ValueError(
        "Cannot use --motion-dir with single motion command. Use --motion-file instead."
      )
    motion_cmd.motion_dir = cfg.motion_dir
    motion_cmd.motion_name_pattern = cfg.motion_name_pattern
    return None

  if isinstance(motion_cmd, MultiMotionCommandCfg):
    raise ValueError(
      "Must provide --wandb-entity-project, or --registry-name, "
      "or --motion-file or --motion-dir for multi-motion tracking tasks."
    )
  raise ValueError("Must provide --registry-name or --motion-file for tracking tasks.")


def _configure_cuda_environment(selected_gpus: list[int] | None) -> None:
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"


def run_train(
  task_id: str, cfg: TrainConfig, log_dir: Path, motion_name: str | None = None
) -> None:
  device, seed, rank = _resolve_training_runtime(cfg)

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  motion_cmd = _get_tracking_motion_cmd(cfg.env)
  registry_name = (
    _configure_tracking_motion_source(cfg, motion_cmd)
    if motion_cmd is not None
    else None
  )

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    # # Initialize wandb with motion name before runner initializes it
    # # This ensures the wandb run name is the motion name, not the log directory name
    # if motion_name and cfg.agent.logger == "wandb":
    #   import wandb
    #   if wandb.run is None:
    #     wandb.init(
    #       project=cfg.agent.wandb_project,
    #       name=motion_name,
    #       config={},
    #       dir=str(log_dir.parent),  # Set dir to motion_name directory, not timestamp subdirectory
    #     )

  env = ManagerBasedRlEnv(
    cfg=cfg.env, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = log_dir.parent  # Go up from specific run dir to experiment dir.

  resume_path: Path | None = None
  if cfg.agent.resume:
    if cfg.wandb_run_path is not None:
      # Load checkpoint from W&B.
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      if rank == 0:
        run_id = resume_path.parent.name
        checkpoint_name = resume_path.name
        cached_str = "cached" if was_cached else "downloaded"
        print(
          f"[INFO]: Loading checkpoint from W&B: {checkpoint_name} "
          f"(run: {run_id}, {cached_str})"
        )
    else:
      # Load checkpoint from local filesystem.
      resume_path = get_checkpoint_path(
        log_root_path, cfg.agent.load_run, cfg.agent.load_checkpoint
      )

  # Only record videos on rank 0 to avoid multiple workers writing to the same files.
  if cfg.video and rank == 0:
    env = VideoRecorder(
      env,
      video_folder=Path(log_dir) / "videos" / "train",
      step_trigger=lambda step: step % cfg.video_interval == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )
    print("[INFO] Recording videos during training.")

  env = RslRlVecEnvWrapper(env, clip_actions=cfg.agent.clip_actions)

  agent_cfg = asdict(cfg.agent)
  env_cfg = asdict(cfg.env)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = MjlabOnPolicyRunner

  runner_kwargs = {}
  if motion_cmd is not None:
    runner_kwargs["registry_name"] = registry_name

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

  add_wandb_tags(cfg.agent.wandb_tags)
  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg)

  runner.learn(
    num_learning_iterations=cfg.agent.max_iterations, init_at_random_ep_len=True
  )

  env.close()


def launch_training(task_id: str, args: TrainConfig | None = None):
  args = args or TrainConfig.from_task(task_id)

  # Extract motion name for tracking tasks to use as run_name
  # This ensures the log directory name (and thus wandb run name) includes the motion name
  from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiMotionCommandCfg

  is_tracking_task = (
    args.env.commands is not None
    and "motion" in args.env.commands
    and isinstance(
      args.env.commands["motion"], (MotionCommandCfg, MultiMotionCommandCfg)
    )
  )

  motion_name: str | None = None
  if is_tracking_task and not args.agent.run_name:
    assert args.env.commands is not None
    motion_cmd = args.env.commands["motion"]
    assert isinstance(motion_cmd, (MotionCommandCfg, MultiMotionCommandCfg))

    # Extract motion name from different sources
    if isinstance(motion_cmd, MotionCommandCfg):
      if args.registry_name:
        # Extract from registry name (e.g., "entity/project/motion_name:latest" -> "motion_name")
        registry_name = cast(str, args.registry_name)
        if ":" not in registry_name:
          registry_name = registry_name + ":latest"
        motion_name = registry_name.split("/")[-1].split(":")[0]
        args.agent.run_name = motion_name
      elif args.motion_file:
        # Extract from file path (e.g., "/path/to/motion_name/motion.npz" -> "motion_name")
        motion_file_path = Path(args.motion_file)
        motion_name = (
          motion_file_path.parent.name
          if motion_file_path.parent.name
          else motion_file_path.stem
        )
        args.agent.run_name = motion_name
    elif isinstance(motion_cmd, MultiMotionCommandCfg):
      # For MultiMotionCommandCfg, extract motion name from pattern or motion_dir
      if (
        args.motion_name_pattern
        and len(args.motion_name_pattern) == 1
        and args.motion_name_pattern[0] != ".*"
      ):
        # Use the single pattern as motion name
        motion_name = args.motion_name_pattern[0]
        # Don't use motion_name_pattern for run_name if it's too long or contains multiple patterns
        if "|" not in motion_name:
          args.agent.run_name = motion_name
        # Otherwise, use the wandb project name
        elif args.wandb_entity_project is not None:
          _, wandb_project = get_wandb_entity_and_project(args.wandb_entity_project)
          args.agent.run_name = wandb_project

      elif args.motion_dir:
        # Extract from motion_dir path (e.g., "/path/to/motion_name" -> "motion_name")
        motion_dir_path = Path(args.motion_dir)
        motion_name = motion_dir_path.name
        args.agent.run_name = motion_name
      elif args.registry_name:
        # Extract from registry name (e.g., "entity/project/motion_name:latest" -> "motion_name")
        registry_name = cast(str, args.registry_name)
        if ":" not in registry_name:
          registry_name = registry_name + ":latest"
        motion_name = registry_name.split("/")[-1].split(":")[0]
        args.agent.run_name = motion_name

  # Create log directory once before launching workers.
  # log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  # log_root_path = os.environ.get("MJLAB_LOG_PATH", None)
  # assert log_root_path is not None, "Environment variable MJLAB_LOG_PATH must be set."
  # log_root_path = Path(log_root_path) / args.agent.experiment_name
  # log_root_path.resolve()
  # log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  # if args.agent.run_name:
  #   log_dir_name += f"_{args.agent.run_name}"
  # log_dir = log_root_path / log_dir_name
  log_dir = get_log_dir(
    experiment_name=args.agent.experiment_name, run_name=args.agent.run_name
  )

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  _configure_cuda_environment(selected_gpus)

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir, motion_name)
  else:
    # Multi-GPU: use torchrunx.
    import torchrunx

    # torchrunx redirects stdout to logging.
    logging.basicConfig(level=logging.INFO)

    # Configure torchrunx logging directory.
    # Priority: 1) existing env var, 2) user flag, 3) default to {log_dir}/torchrunx.
    if "TORCHRUNX_LOG_DIR" not in os.environ:
      if args.torchrunx_log_dir is not None:
        # User specified a value via flag (could be "" to disable).
        os.environ["TORCHRUNX_LOG_DIR"] = args.torchrunx_log_dir
      else:
        # Default: put logs in training directory.
        os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

    print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
    torchrunx.Launcher(
      hostnames=["localhost"],
      workers_per_host=num_gpus,
      backend=None,  # Let rsl_rl handle process group initialization.
      copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
    ).run(run_train, task_id, args, log_dir, motion_name)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, args = parse_choice_and_dataclass(
    all_tasks,
    TrainConfig,
    default_factory=TrainConfig.from_task,
    prog=sys.argv[0],
    description=__doc__,
  )

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
