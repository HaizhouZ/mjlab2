"""Script to train RL agent with RSL-RL."""

import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiMotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import download_motions_from_wandb
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainConfig:
  env: ManagerBasedRlEnvCfg
  agent: RslRlOnPolicyRunnerCfg
  registry_name: str | None = None
  motion_file: str | None = None
  motion_dir: str | None = None
  wandb_entity_project: str | None = None
  device: str = "cuda:0"
  video: bool = False
  video_length: int = 200
  video_interval: int = 2000
  enable_nan_guard: bool = False
  torchrunx_log_dir: str | None = None
  wandb_run_path: str | None = None
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])

  def get_wandb_entity_and_project(self) -> tuple[str | None, str | None]:
    """Get wandb entity and project from combined format."""
    if self.wandb_entity_project:
      parts = self.wandb_entity_project.split("/", 1)
      if len(parts) == 2:
        return parts[0], parts[1]
      else:
        raise ValueError(
          f"wandb_entity_project must be in format 'entity/project', got: {self.wandb_entity_project}"
        )
    return None, None

  @staticmethod
  def from_task(task_id: str) -> "TrainConfig":
    env_cfg = load_env_cfg(task_id)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    return TrainConfig(env=env_cfg, agent=agent_cfg)


def run_train(task_id: str, cfg: TrainConfig, log_dir: Path) -> None:
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = cfg.agent.seed + local_rank

  configure_torch_backends()

  cfg.agent.seed = seed
  cfg.env.seed = seed

  print(f"[INFO] Training with: device={device}, seed={seed}, rank={rank}")

  registry_name: str | None = None

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = (
    cfg.env.commands is not None
    and "motion" in cfg.env.commands
    and isinstance(
      cfg.env.commands["motion"], (MotionCommandCfg, MultiMotionCommandCfg)
    )
  )

  if is_tracking_task:
    assert cfg.env.commands is not None
    motion_cmd = cfg.env.commands["motion"]
    assert isinstance(motion_cmd, (MotionCommandCfg, MultiMotionCommandCfg))

    wandb_entity, wandb_project = cfg.get_wandb_entity_and_project()
    if isinstance(motion_cmd, MultiMotionCommandCfg) and wandb_entity and wandb_project:
      # Use wandb registry to download multiple motions
      print(f"[INFO] Downloading motions from wandb: {wandb_entity}/{wandb_project}")
      motions_dir = download_motions_from_wandb(
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        artifact_type="motions",
      )
      motion_cmd.motion_dir = str(motions_dir)
      motion_cmd.traj_name_patterns = [".*"]
    elif cfg.registry_name:
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cast(str, cfg.registry_name)
      if ":" not in registry_name:
        registry_name = registry_name + ":latest"
      import wandb

      api = wandb.Api()
      artifact = api.artifact(registry_name)
      if isinstance(motion_cmd, MotionCommandCfg):
        motion_cmd.motion_file = str(Path(artifact.download()) / "motion.npz")
      elif isinstance(motion_cmd, MultiMotionCommandCfg):
        motion_dir = str(Path(artifact.download()) / "motions")
        motion_cmd.motion_dir = motion_dir
        motion_cmd.traj_name_patterns = [".*"]
    elif cfg.motion_file is not None:
      # motion_file provided via CLI: --motion-file /path/to/motion.npz
      print(f"[INFO] Using local motion file: {cfg.motion_file}")
      if isinstance(motion_cmd, MotionCommandCfg):
        motion_cmd.motion_file = cfg.motion_file
      elif isinstance(motion_cmd, MultiMotionCommandCfg):
        motion_dir = cfg.motion_dir
        if motion_dir is None:
          raise ValueError("Must provide --motion-dir for multi-motion tracking tasks.")
        motion_cmd.motion_dir = motion_dir
        motion_cmd.traj_name_patterns = [".*"]
    elif cfg.motion_dir is not None:
      # motion_dir provided via CLI: --motion-dir /path/to/motion/dir
      print(f"[INFO] Using local motion directory: {cfg.motion_dir}")
      if isinstance(motion_cmd, MotionCommandCfg):
        raise ValueError(
          "Cannot use --motion-dir with single motion command. Use --motion-file instead."
        )
      elif isinstance(motion_cmd, MultiMotionCommandCfg):
        motion_cmd.motion_dir = cfg.motion_dir
        motion_cmd.traj_name_patterns = [".*"]
    else:
      if isinstance(motion_cmd, MultiMotionCommandCfg):
        raise ValueError(
          "Must provide --wandb-entity-project, or --registry-name, "
          "or --motion-file or --motion-dir for multi-motion tracking tasks."
        )
      else:
        raise ValueError(
          "Must provide --registry-name or --motion-file for tracking tasks."
        )

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    cfg.env.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {cfg.env.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")

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
    runner_cls = OnPolicyRunner

  runner_kwargs = {}
  if is_tracking_task:
    runner_kwargs["registry_name"] = registry_name

  runner = runner_cls(env, agent_cfg, str(log_dir), device, **runner_kwargs)

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

  # Create log directory once before launching workers.
  log_root_path = Path("logs") / "rsl_rl" / args.agent.experiment_name
  log_root_path.resolve()
  log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
  if args.agent.run_name:
    log_dir_name += f"_{args.agent.run_name}"
  log_dir = log_root_path / log_dir_name

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(args.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  if num_gpus <= 1:
    # CPU or single GPU: run directly without torchrunx.
    run_train(task_id, args, log_dir)
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
    ).run(run_train, task_id, args, log_dir)


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, remaining_args = tyro.cli(
    tyro.extras.literal_type_from_choices(all_tasks),
    add_help=False,
    return_unknown_args=True,
  )

  args = tyro.cli(
    TrainConfig,
    args=remaining_args,
    default=TrainConfig.from_task(chosen_task),
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )
  del remaining_args

  launch_training(task_id=chosen_task, args=args)


if __name__ == "__main__":
  main()
