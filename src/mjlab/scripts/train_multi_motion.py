"""Script to train separate policies for multiple motions using MotionCommandCfg.

This script trains a different policy for each motion provided via wandb registry names or motion directory.
Each training run is logged separately to wandb.

Usage:
    # Train multiple motions from wandb registry
    uv run src/mjlab/scripts/train_multi_motion.py Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation\
        --registry-names wandb-registry-motions/motion1 wandb-registry-motions/motion2

    # Train all motions from a local directory
    uv run src/mjlab/scripts/train_multi_motion.py Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation\
        --motion-dir motions/output/diverse \
        --traj-name-patterns ".*"

    # Override environment and agent configs
    uv run src/mjlab/scripts/train_multi_motion.py Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation\
        --motion-dir motions/output/diverse \
        --env.scene.num_envs 4096 \
        --agent.max-iterations 5000

    # With custom experiment name prefix
    uv run src/mjlab/scripts/train_multi_motion.py Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation\
        --registry-names wandb-registry-motions/motion1 wandb-registry-motions/motion2 \
        --experiment-name-prefix my_experiment

    # With video recording
    uv run src/mjlab/scripts/train_multi_motion.py Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation\
        --registry-names wandb-registry-motions/motion1 \
        --video --video-interval 1000

    # Using multiple GPUs
    CUDA_VISIBLE_DEVICES=0,1 uv run src/mjlab/scripts/train_multi_motion.py Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation\
        --registry-names wandb-registry-motions/motion1 wandb-registry-motions/motion2 \
        --gpu-ids all

After training, to play a policy:
    # Using wandb run path
    uv run play Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation \
        --wandb-run-path your-entity/omniretarget_motions/run-id

    # Using checkpoint file
    uv run play Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation \
        --checkpoint-file logs/rsl_rl/experiment_name/run_id/model_*.pt \
        --motion-file /path/to/motion.npz
"""

import logging
import os
import sys
from copy import deepcopy
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Literal, cast

import tyro
import wandb
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiMotionCommandCfg
from mjlab.utils.gpu import select_gpus
from mjlab.utils.os import dump_yaml, get_checkpoint_path, get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wrappers import VideoRecorder


@dataclass(frozen=True)
class TrainMultiMotionConfig:
  """Configuration for training multiple motion policies."""

  registry_names: list[str] = field(default_factory=list)
  """List of wandb registry names for motions (e.g., ['wandb-registry-motions/motion1', ...]).
  Each motion will get its own trained policy. Mutually exclusive with motion_dir."""
  motion_dir: str | None = None
  """Directory containing motion trajectories. Each subdirectory should contain motion.npz.
  Mutually exclusive with registry_names. Structure: <motion_dir>/<traj_name>/motion.npz"""
  motion_name_pattern: list[str] = field(default_factory=lambda: [".*"])
  """Regex patterns to match trajectory names when using motion_dir. Default matches all."""
  task_id: str = "Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation"
  """Task ID to use for training. Must be a tracking task with MotionCommandCfg."""
  env: ManagerBasedRlEnvCfg | None = None
  """Environment configuration. If None, uses default from task. Can be overridden with --env.* flags."""
  agent: RslRlOnPolicyRunnerCfg | None = None
  """Agent configuration. If None, uses default from task. Can be overridden with --agent.* flags."""
  device: str = "cuda:0"
  """Device to use for training."""
  video: bool = False
  """Whether to record videos during training."""
  video_length: int = 200
  """Length of videos to record."""
  video_interval: int = 2000
  """Interval between video recordings."""
  enable_nan_guard: bool = False
  """Whether to enable NaN guard."""
  torchrunx_log_dir: str | None = None
  """Directory for torchrunx logs."""
  wandb_run_path: str | None = None
  """Wandb run path for resuming training."""
  gpu_ids: list[int] | Literal["all"] | None = field(default_factory=lambda: [0])
  """GPU IDs to use for training."""
  experiment_name: str = "multi_motion"
  """Experiment name."""
  run_name_suffix: str | None = None
  """Optional suffix to add to run names."""


def run_train_single_motion_from_registry(
  task_id: str,
  registry_name: str,
  motion_name: str,
  cfg: TrainMultiMotionConfig,
  log_dir: Path,
) -> None:
  """Train a single policy for one motion from wandb registry.

  Args:
    task_id: Task ID to use for training.
    registry_name: Wandb registry name for the motion (e.g., 'wandb-registry-motions/motion1:latest').
    motion_name: Name of the motion (used for logging/experiment naming).
    cfg: Training configuration.
    log_dir: Directory to save logs.
  """
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed if cfg.agent else 42
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = (cfg.agent.seed if cfg.agent else 42) + local_rank

  configure_torch_backends()

  # Load environment and agent configs
  env_cfg = cfg.env if cfg.env else load_env_cfg(task_id)
  agent_cfg = cfg.agent if cfg.agent else load_rl_cfg(task_id)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  agent_cfg.seed = seed
  env_cfg.seed = seed

  print(
    f"[INFO] Training motion '{motion_name}' with: device={device}, seed={seed}, rank={rank}"
  )

  # Ensure registry name has alias
  registry_name_with_alias = registry_name
  if ":" not in registry_name_with_alias:
    registry_name_with_alias = registry_name_with_alias + ":latest"

  # Download motion from wandb registry
  api = wandb.Api()
  artifact = api.artifact(registry_name_with_alias)
  motion_file_path = str(Path(artifact.download()) / "motion.npz")

  # Configure motion command
  assert env_cfg.commands is not None
  assert "motion" in env_cfg.commands
  motion_cmd = env_cfg.commands["motion"]
  # assert isinstance(motion_cmd, MotionCommandCfg), (
  #   f"Task {task_id} must use MotionCommandCfg, not MultiMotionCommandCfg"
  # )
  # breakpoint()
  # Extract motion name from different sources
  if registry_name:
    # Extract from registry name (e.g., "entity/project/motion_name:latest" -> "motion_name")
    registry_name = cast(str, registry_name)
    if ":" not in registry_name:
      registry_name = registry_name + ":latest"
    motion_name = registry_name.split("/")[-1].split(":")[0]
    agent_cfg.run_name = motion_name
  if isinstance(motion_cmd, MotionCommandCfg):
    motion_cmd.motion_file = motion_file_path
  elif isinstance(motion_cmd, MultiMotionCommandCfg):
    motion_cmd.motion_dir = str(Path(motion_file_path).parent.parent)
    motion_cmd.motion_name_pattern = [motion_name]

  print(f"[INFO] Using motion file from wandb: {motion_file_path}")
  print(f"[INFO] Registry name: {registry_name_with_alias}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    env_cfg.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {env_cfg.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    # Initialize wandb with motion name before runner initializes it
    # This ensures the wandb run name is the motion name, not the log directory name
    if agent_cfg.logger == "wandb" and wandb.run is None:
      wandb_run_name = agent_cfg.run_name if agent_cfg.run_name else motion_name
      wandb.init(
        project=agent_cfg.wandb_project,
        name=wandb_run_name,
        config={},
        dir=str(
          log_dir.parent
        ),  # Set dir to motion_name directory, not timestamp subdirectory
      )

  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = (
    log_dir.parent.parent
  )  # Go up from timestamp dir to motion_name dir to experiment dir.

  resume_path: Path | None = None
  if agent_cfg.resume:
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
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
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

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  agent_cfg_dict = asdict(agent_cfg)
  env_cfg_dict = asdict(env_cfg)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = OnPolicyRunner

  # Pass registry name to runner for linking artifacts (only for tracking tasks)
  runner_kwargs = {}
  if runner_cls.__name__ == "MotionTrackingOnPolicyRunner":
    runner_kwargs["registry_name"] = registry_name_with_alias

  runner = runner_cls(env, agent_cfg_dict, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg_dict)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg_dict)

  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )

  # Finish wandb run to allow next motion to start fresh
  if wandb.run is not None:
    wandb.finish()

  env.close()


def run_train_single_motion_from_file(
  task_id: str,
  motion_file: str,
  motion_name: str,
  cfg: TrainMultiMotionConfig,
  log_dir: Path,
) -> None:
  """Train a single policy for one motion from local file.

  Args:
    task_id: Task ID to use for training.
    motion_file: Path to motion.npz file.
    motion_name: Name of the motion (used for logging/experiment naming).
    cfg: Training configuration.
    log_dir: Directory to save logs.
  """
  cuda_visible = os.environ.get("CUDA_VISIBLE_DEVICES", "")
  if cuda_visible == "":
    device = "cpu"
    seed = cfg.agent.seed if cfg.agent else 42
    rank = 0
  else:
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    rank = int(os.environ.get("RANK", "0"))
    # Set EGL device to match the CUDA device.
    os.environ["MUJOCO_EGL_DEVICE_ID"] = str(local_rank)
    device = f"cuda:{local_rank}"
    # Set seed to have diversity in different processes.
    seed = (cfg.agent.seed if cfg.agent else 42) + local_rank

  configure_torch_backends()

  # Load environment and agent configs
  env_cfg = cfg.env if cfg.env else load_env_cfg(task_id)
  agent_cfg = cfg.agent if cfg.agent else load_rl_cfg(task_id)
  assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)

  agent_cfg.seed = seed
  env_cfg.seed = seed

  print(
    f"[INFO] Training motion '{motion_name}' with: device={device}, seed={seed}, rank={rank}"
  )

  # Configure motion command
  assert env_cfg.commands is not None
  assert "motion" in env_cfg.commands
  motion_cmd = env_cfg.commands["motion"]
  # assert isinstance(motion_cmd, MotionCommandCfg), (
  #   f"Task {task_id} must use MotionCommandCfg, not MultiMotionCommandCfg"
  # )
  motion_cmd.motion_file = motion_file

  print(f"[INFO] Using motion file: {motion_file}")

  # Enable NaN guard if requested.
  if cfg.enable_nan_guard:
    env_cfg.sim.nan_guard.enabled = True
    print(f"[INFO] NaN guard enabled, output dir: {env_cfg.sim.nan_guard.output_dir}")

  if rank == 0:
    print(f"[INFO] Logging experiment in directory: {log_dir}")
    # Initialize wandb with motion name before runner initializes it
    # This ensures the wandb run name is the motion name, not the log directory name
    if agent_cfg.logger == "wandb" and wandb.run is None:
      wandb_run_name = agent_cfg.run_name if agent_cfg.run_name else motion_name
      wandb.init(
        project=agent_cfg.wandb_project,
        name=wandb_run_name,
        config={},
        dir=str(
          log_dir.parent
        ),  # Set dir to motion_name directory, not timestamp subdirectory
      )

  env = ManagerBasedRlEnv(
    cfg=env_cfg, device=device, render_mode="rgb_array" if cfg.video else None
  )

  log_root_path = (
    log_dir.parent.parent
  )  # Go up from timestamp dir to motion_name dir to experiment dir.

  resume_path: Path | None = None
  if agent_cfg.resume:
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
        log_root_path, agent_cfg.load_run, agent_cfg.load_checkpoint
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

  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  agent_cfg_dict = asdict(agent_cfg)
  env_cfg_dict = asdict(env_cfg)

  runner_cls = load_runner_cls(task_id)
  if runner_cls is None:
    runner_cls = OnPolicyRunner

  # Pass registry name to runner for linking artifacts (only for tracking tasks)
  runner_kwargs = {}
  if runner_cls.__name__ == "MotionTrackingOnPolicyRunner":
    # No registry name for local files
    runner_kwargs["registry_name"] = None

  runner = runner_cls(env, agent_cfg_dict, str(log_dir), device, **runner_kwargs)

  runner.add_git_repo_to_log(__file__)
  if resume_path is not None:
    print(f"[INFO]: Loading model checkpoint from: {resume_path}")
    runner.load(str(resume_path))

  # Only write config files from rank 0 to avoid race conditions.
  if rank == 0:
    dump_yaml(log_dir / "params" / "env.yaml", env_cfg_dict)
    dump_yaml(log_dir / "params" / "agent.yaml", agent_cfg_dict)

  # Log motion file as artifact to wandb if using local file (only on rank 0)
  # Note: wandb may not be initialized until learn() is called, so we'll try to log it
  # after learn() starts. For now, we'll attempt it here, but it may need to be logged
  # after the first save (similar to how registry artifacts are linked).
  if rank == 0:
    motion_file_path = Path(motion_file)
    if motion_file_path.exists() and wandb.run is not None:
      # Log the motion file as an artifact
      artifact_name = f"{motion_name}_motion"
      print(f"[INFO]: Logging motion file as artifact: {artifact_name}")
      try:
        _ = wandb.run.log_artifact(
          artifact_or_path=str(motion_file_path),
          name=artifact_name,
          type="motions",
        )
        print(f"[INFO]: Motion artifact logged: {artifact_name}")
      except Exception as e:
        print(f"[WARN]: Failed to log motion artifact: {e}")
        print("[WARN]: Motion artifact will need to be provided manually when playing")

  runner.learn(
    num_learning_iterations=agent_cfg.max_iterations, init_at_random_ep_len=True
  )

  # Finish wandb run to allow next motion to start fresh
  if wandb.run is not None:
    wandb.finish()

  env.close()


def find_motion_trajectories(
  motion_dir: str, motion_name_pattern: list[str]
) -> list[tuple[str, str]]:
  """Find all motion trajectories in a directory.

  Args:
    motion_dir: Base directory containing trajectory subdirectories.
    motion_name_pattern: List of regex patterns to match trajectory names.

  Returns:
    List of (trajectory_name, motion_file_path) tuples.
  """
  import re

  motion_dir_path = Path(motion_dir)
  if not motion_dir_path.exists():
    raise ValueError(f"Motion directory does not exist: {motion_dir}")

  traj_files = []
  for pattern in motion_name_pattern:
    regex = re.compile(pattern)
    matching_dirs = [
      d
      for d in motion_dir_path.iterdir()
      if d.is_dir() and regex.match(d.name) and (d / "motion.npz").exists()
    ]
    for traj_dir in matching_dirs:
      motion_file = traj_dir / "motion.npz"
      traj_files.append((traj_dir.name, str(motion_file)))

  # Remove duplicates and sort for reproducibility
  traj_files = sorted(set(traj_files), key=lambda x: x[0])

  if not traj_files:
    raise ValueError(
      f"No matching trajectories found in {motion_dir} with patterns {motion_name_pattern}"
    )

  return traj_files


def launch_training_multi_motion(cfg: TrainMultiMotionConfig) -> None:
  """Launch training for multiple motions.

  Args:
    cfg: Configuration for training multiple motion policies.
  """
  # Import tasks to populate registry
  import mjlab.tasks  # noqa: F401

  # Validate that either registry_names or motion_dir is provided
  if not cfg.registry_names and not cfg.motion_dir:
    raise ValueError(
      "Must provide either --registry-names or --motion-dir for training."
    )
  if cfg.registry_names and cfg.motion_dir:
    raise ValueError(
      "Cannot provide both --registry-names and --motion-dir. Use one or the other."
    )

  # Load configs if not provided (will be used as defaults for each motion)
  env_cfg_default = cfg.env if cfg.env else load_env_cfg(cfg.task_id)
  agent_cfg_default = cfg.agent if cfg.agent else load_rl_cfg(cfg.task_id)
  assert isinstance(agent_cfg_default, RslRlOnPolicyRunnerCfg)

  # Prepare list of motions to train
  motions_to_train: list[
    tuple[str, str | None, str | None]
  ] = []  # (name, registry_name, motion_file)
  if cfg.motion_dir:
    # Find all trajectories in the motion directory
    traj_files = find_motion_trajectories(cfg.motion_dir, cfg.motion_name_pattern)
    print(f"[INFO] Found {len(traj_files)} trajectories in {cfg.motion_dir}")
    for traj_name, motion_file in traj_files:
      motions_to_train.append((traj_name, None, motion_file))
  else:
    # Use registry names
    for registry_name in cfg.registry_names:
      # Extract motion name from registry name
      motion_name = registry_name.split("/")[-1].split(":")[0]
      motions_to_train.append((motion_name, registry_name, None))

  # Select GPUs based on CUDA_VISIBLE_DEVICES and user specification.
  selected_gpus, num_gpus = select_gpus(cfg.gpu_ids)

  # Set environment variables for all modes.
  if selected_gpus is None:
    os.environ["CUDA_VISIBLE_DEVICES"] = ""
  else:
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, selected_gpus))
  os.environ["MUJOCO_GL"] = "egl"

  # Train each motion sequentially
  for i, (motion_name, registry_name, motion_file) in enumerate(motions_to_train):
    print("\n" + "=" * 80)
    print(f"Training motion {i + 1}/{len(motions_to_train)}: {motion_name}")
    if registry_name:
      print(f"Registry: {registry_name}")
    if motion_file:
      print(f"File: {motion_file}")
    print("=" * 80)

    # Create experiment name
    experiment_name = f"{cfg.experiment_name}"

    # Create a config for this specific motion
    # Deep copy configs to avoid mutation between motions
    motion_agent_cfg = deepcopy(agent_cfg_default)
    motion_agent_cfg.run_name = motion_name

    # Create nested log directory structure: logs/rsl_rl/{experiment_name}/{motion_name}/{timestamp}
    # This prevents overwriting while keeping motion name for wandb
    log_root_path = Path("logs") / "rsl_rl" / experiment_name / motion_name
    log_root_path.resolve()
    log_dir_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if cfg.run_name_suffix:
      log_dir_name = f"{log_dir_name}_{cfg.run_name_suffix}"
    log_dir = log_root_path / log_dir_name

    # Add "pd" suffix if PDTargets is in task name
    if "PDTargets" in cfg.task_id:
      motion_agent_cfg.run_name += "_pd"
    elif "Default" in cfg.task_id:
      motion_agent_cfg.run_name += "_default"

    motion_cfg = TrainMultiMotionConfig(
      registry_names=[registry_name] if registry_name else [],
      motion_dir=None,
      motion_name_pattern=cfg.motion_name_pattern,
      task_id=cfg.task_id,
      env=deepcopy(env_cfg_default),
      agent=motion_agent_cfg,
      device=cfg.device,
      video=cfg.video,
      video_length=cfg.video_length,
      video_interval=cfg.video_interval,
      enable_nan_guard=cfg.enable_nan_guard,
      torchrunx_log_dir=cfg.torchrunx_log_dir,
      wandb_run_path=cfg.wandb_run_path,
      gpu_ids=cfg.gpu_ids,
      experiment_name=cfg.experiment_name,
    )

    if num_gpus <= 1:
      # CPU or single GPU: run directly without torchrunx.
      if registry_name:
        run_train_single_motion_from_registry(
          cfg.task_id, registry_name, motion_name, motion_cfg, log_dir
        )
      else:
        assert motion_file is not None
        run_train_single_motion_from_file(
          cfg.task_id, motion_file, motion_name, motion_cfg, log_dir
        )
    else:
      # Multi-GPU: use torchrunx.
      import torchrunx

      # torchrunx redirects stdout to logging.
      logging.basicConfig(level=logging.INFO)

      # Configure torchrunx logging directory.
      if "TORCHRUNX_LOG_DIR" not in os.environ:
        if motion_cfg.torchrunx_log_dir is not None:
          os.environ["TORCHRUNX_LOG_DIR"] = motion_cfg.torchrunx_log_dir
        else:
          os.environ["TORCHRUNX_LOG_DIR"] = str(log_dir / "torchrunx")

      print(f"[INFO] Launching training with {num_gpus} GPUs", flush=True)
      if registry_name:
        torchrunx.Launcher(
          hostnames=["localhost"],
          workers_per_host=num_gpus,
          backend=None,  # Let rsl_rl handle process group initialization.
          copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(
          run_train_single_motion_from_registry,
          cfg.task_id,
          registry_name,
          motion_name,
          motion_cfg,
          log_dir,
        )
      else:
        assert motion_file is not None
        torchrunx.Launcher(
          hostnames=["localhost"],
          workers_per_host=num_gpus,
          backend=None,  # Let rsl_rl handle process group initialization.
          copy_env_vars=torchrunx.DEFAULT_ENV_VARS_FOR_COPY + ("MUJOCO*",),
        ).run(
          run_train_single_motion_from_file,
          cfg.task_id,
          motion_file,
          motion_name,
          motion_cfg,
          log_dir,
        )

    print(
      f"\n[INFO] Completed training for motion {i + 1}/{len(motions_to_train)}: {motion_name}"
    )
    print(f"[INFO] Log directory: {log_dir}")

  print("\n" + "=" * 80)
  print(f"Completed training all {len(motions_to_train)} motions!")
  print("=" * 80)


def main():
  """Main entry point."""
  # Import tasks to populate registry
  import mjlab.tasks  # noqa: F401
  from mjlab.tasks.registry import list_tasks

  # Parse first argument to choose the task (optional, defaults to Box-No-State-Estimation)
  all_tasks = list_tasks()
  default_task = "Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation"

  # Try to parse task ID from command line (first positional arg)
  # Check if first arg is a valid task ID (not a flag)
  if len(sys.argv) > 1 and not sys.argv[1].startswith("-") and sys.argv[1] in all_tasks:
    chosen_task, remaining_args = tyro.cli(
      tyro.extras.literal_type_from_choices(all_tasks),
      add_help=False,
      return_unknown_args=True,
    )
  else:
    # Use default task if first arg is a flag or not a valid task
    chosen_task = default_task
    remaining_args = sys.argv[1:] if len(sys.argv) > 1 else []

  # Load default configs from task
  default_env_cfg = load_env_cfg(chosen_task)
  default_agent_cfg = load_rl_cfg(chosen_task)
  assert isinstance(default_agent_cfg, RslRlOnPolicyRunnerCfg)

  # Create default config
  default_cfg = TrainMultiMotionConfig(
    registry_names=[],  # Must provide either registry_names or motion_dir
    motion_dir=None,
    motion_name_pattern=[".*"],
    task_id=chosen_task,
    env=default_env_cfg,
    agent=default_agent_cfg,
  )

  args = tyro.cli(
    TrainMultiMotionConfig,
    args=remaining_args,
    default=default_cfg,
    prog=sys.argv[0] + f" {chosen_task}",
    config=(
      tyro.conf.AvoidSubcommands,
      tyro.conf.FlagConversionOff,
    ),
  )

  # Override task_id if it was provided as positional arg
  if args.task_id != chosen_task:
    # User might have explicitly set --task-id, use that
    chosen_task = args.task_id
    # Reload configs if task changed
    if args.env is None or args.agent is None:
      args = TrainMultiMotionConfig(
        registry_names=args.registry_names,
        task_id=chosen_task,
        env=args.env if args.env else load_env_cfg(chosen_task),
        agent=args.agent if args.agent else load_rl_cfg(chosen_task),
        device=args.device,
        video=args.video,
        video_length=args.video_length,
        video_interval=args.video_interval,
        enable_nan_guard=args.enable_nan_guard,
        torchrunx_log_dir=args.torchrunx_log_dir,
        wandb_run_path=args.wandb_run_path,
        gpu_ids=args.gpu_ids,
        experiment_name=args.experiment_name,
        run_name_suffix=args.run_name_suffix,
      )

  launch_training_multi_motion(args)


if __name__ == "__main__":
  main()
