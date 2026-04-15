"""Script to log qpos and qvel for one episode using N environments."""

import sys
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv
from mjlab.rl import RslRlVecEnvWrapper

# Import PlayConfig from play.py
from mjlab.scripts.play import PlayConfig
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiMotionCommandCfg
from mjlab.tasks.tracking.mdp.commands import MotionCommand
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner
from mjlab.utils import parse_choice_and_dataclass
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends


def run_log_qpos_qvel(task: str, cfg: PlayConfig, output_file: str | None = None):
  """Run one episode and log concatenated qpos and qvel.

  Args:
    task: Task name
    cfg: PlayConfig configuration
    output_file: Output file path (default: qpos_qvel_log.npz)
  """
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = load_env_cfg(task, play=True)
  agent_cfg = load_rl_cfg(task)

  DUMMY_MODE = cfg.agent in {"zero", "random"}
  TRAINED_MODE = not DUMMY_MODE

  # Check if this is a tracking task by checking for motion command.
  is_tracking_task = (
    env_cfg.commands is not None
    and "motion" in env_cfg.commands
    and isinstance(
      env_cfg.commands["motion"], (MotionCommandCfg, MultiMotionCommandCfg)
    )
  )

  if is_tracking_task:
    assert env_cfg.commands is not None
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, (MotionCommandCfg, MultiMotionCommandCfg))

    if DUMMY_MODE:
      if not cfg.registry_name:
        raise ValueError(
          "Tracking tasks require `registry_name` when using dummy agents."
        )
      # Check if the registry name includes alias, if not, append ":latest".
      registry_name = cfg.registry_name
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
        motion_cmd.motion_name_pattern = [".*"]
    else:
      if isinstance(motion_cmd, MotionCommandCfg) and cfg.motion_file is not None:
        print(f"[INFO]: Using motion file from CLI: {cfg.motion_file}")
        motion_cmd.motion_file = cfg.motion_file
      elif isinstance(motion_cmd, MultiMotionCommandCfg) and cfg.motion_dir is not None:
        print(f"[INFO]: Using motion directory from CLI: {cfg.motion_dir}")
        motion_dir = cfg.motion_dir
        motion_cmd.motion_dir = motion_dir
        motion_cmd.motion_name_pattern = [".*"]
      else:
        import wandb

        api = wandb.Api()
        if cfg.wandb_run_path is None and cfg.checkpoint_file is not None:
          raise ValueError(
            "Tracking tasks require `motion_file` when using `checkpoint_file`, "
            "or provide `wandb_run_path` so the motion artifact can be resolved."
          )
        if cfg.wandb_run_path is not None:
          wandb_run = api.run(str(cfg.wandb_run_path))
          # Check both used_artifacts (linked via use_artifact) and logged_artifacts (logged directly)
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            # Also check logged_artifacts in case the artifact was logged directly
            art = next(
              (a for a in wandb_run.logged_artifacts() if a.type == "motions"), None
            )
          if art is None:
            # Provide helpful error message with available artifacts
            used_arts = list(wandb_run.used_artifacts())
            logged_arts = list(wandb_run.logged_artifacts())
            all_arts = used_arts + logged_arts
            art_types = [a.type for a in all_arts]
            art_names = [a.name for a in all_arts]

            # Try to find motion file path in wandb config as fallback
            motion_file_from_config = None
            try:
              config = wandb_run.config
              # Check if motion_file is stored in config (might be nested)
              if isinstance(config, dict):
                # Check various possible paths in config
                if "env" in config and isinstance(config["env"], dict):
                  if "commands" in config["env"] and isinstance(
                    config["env"]["commands"], dict
                  ):
                    if "motion" in config["env"]["commands"]:
                      motion_cfg = config["env"]["commands"]["motion"]
                      if isinstance(motion_cfg, dict) and "motion_file" in motion_cfg:
                        motion_file_from_config = motion_cfg["motion_file"]
            except Exception:
              pass  # Ignore errors when reading config

            error_msg = (
              f"No motion artifact (type='motions') found in run {cfg.wandb_run_path}.\n"
              f"Found {len(all_arts)} artifact(s) with types: {set(art_types)}\n"
              f"Artifact names: {art_names}\n"
            )
            if motion_file_from_config:
              error_msg += (
                f"\nFound motion_file in config: {motion_file_from_config}\n"
                f"You can use this file directly with: --motion-file {motion_file_from_config}"
              )
            else:
              error_msg += (
                "\nIf you trained with local motion files, the artifact may not have been logged.\n"
                "Please provide the motion file directly using: --motion-file /path/to/motion.npz"
              )
            raise RuntimeError(error_msg)
          if isinstance(motion_cmd, MotionCommandCfg):
            motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")
          elif isinstance(motion_cmd, MultiMotionCommandCfg):
            motion_dir = str(Path(art.download()) / "motions")
            motion_cmd.motion_dir = motion_dir
            motion_cmd.motion_name_pattern = [".*"]

  log_dir: Path | None = None
  resume_path: Path | None = None
  if TRAINED_MODE:
    log_root_path = (Path("logs") / "rsl_rl" / agent_cfg.experiment_name).resolve()
    if cfg.checkpoint_file is not None:
      resume_path = Path(cfg.checkpoint_file)
      if not resume_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {resume_path}")
      print(f"[INFO]: Loading checkpoint: {resume_path.name}")
    else:
      if cfg.wandb_run_path is None:
        raise ValueError(
          "`wandb_run_path` is required when `checkpoint_file` is not provided."
        )
      resume_path, was_cached = get_wandb_checkpoint_path(
        log_root_path, Path(cfg.wandb_run_path)
      )
      # Extract run_id and checkpoint name from path for display.
      run_id = resume_path.parent.name
      checkpoint_name = resume_path.name
      cached_str = "cached" if was_cached else "downloaded"
      print(
        f"[INFO]: Loading checkpoint: {checkpoint_name} (run: {run_id}, {cached_str})"
      )
    log_dir = resume_path.parent

  if cfg.num_envs is not None:
    env_cfg.scene.num_envs = cfg.num_envs

  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=None)
  env = RslRlVecEnvWrapper(env, clip_actions=agent_cfg.clip_actions)

  if DUMMY_MODE:
    action_shape: tuple[int, ...] = env.unwrapped.action_space.shape  # type: ignore
    if cfg.agent == "zero":

      class PolicyZero:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return torch.zeros(action_shape, device=env.unwrapped.device)

      policy = PolicyZero()
    else:

      class PolicyRandom:
        def __call__(self, obs) -> torch.Tensor:
          del obs
          return 2 * torch.rand(action_shape, device=env.unwrapped.device) - 1

      policy = PolicyRandom()
  else:
    if is_tracking_task:
      runner = MotionTrackingOnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    else:
      runner = OnPolicyRunner(
        env, asdict(agent_cfg), log_dir=str(log_dir), device=device
      )
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

  # Reset environment
  obs, _ = env.reset()

  # Get episode length from environment
  motion_cmd = env.unwrapped.command_manager.get_term("motion")
  if isinstance(motion_cmd, MotionCommand):
    episode_length = motion_cmd.motion.time_step_total
  else:
    raise ValueError("Motion type must be MotionCommand.")
  num_envs = env.unwrapped.num_envs

  # Storage for logged data
  logged_data = []
  logged_time = []

  # Get mj_data reference for dimensions (will be set in first step)
  mj_data = env.unwrapped.sim.data

  print(f"[INFO]: Starting episode logging with {num_envs} environments")
  print(f"[INFO]: Episode length: {episode_length} steps")

  # Run for exactly one episode (episode_length steps)
  for step in range(episode_length):
    # Get action from policy
    with torch.no_grad():
      action = policy(obs)

    # Step environment (wrapper returns: obs, reward, dones, extras)
    obs, _, dones, _ = env.step(action)

    # Log qpos, qvel, and time at each step
    mj_data = env.unwrapped.sim.data
    # Access underlying tensors directly to avoid recursion issues with TorchArray
    qpos_tensor = (
      mj_data.qpos._tensor if hasattr(mj_data.qpos, "_tensor") else mj_data.qpos
    )
    qvel_tensor = (
      mj_data.qvel._tensor if hasattr(mj_data.qvel, "_tensor") else mj_data.qvel
    )
    time_tensor = (
      mj_data.time._tensor if hasattr(mj_data.time, "_tensor") else mj_data.time
    )
    # Reshape time to (num_envs, 1) for concatenation
    time_tensor = (
      time_tensor.unsqueeze(-1) if len(time_tensor.shape) == 1 else time_tensor
    )
    x = torch.cat([qpos_tensor, qvel_tensor], dim=-1)
    logged_data.append(x.cpu().clone())
    logged_time.append(time_tensor.cpu().clone())

    if (step + 1) % 100 == 0:
      print(f"[INFO]: Step {step + 1}/{episode_length}")

  print(f"[INFO]: Episode complete after {episode_length} steps")

  # Stack all logged data: shape will be (episode_length, num_envs, feature_dim)
  logged_array = torch.stack(logged_data, dim=0).numpy()
  logged_time_array = torch.stack(logged_time, dim=0).numpy()

  # Reshape to (num_envs, episode_length, feature_dim)
  logged_array = logged_array.transpose(1, 0, 2)
  # Reshape time to (num_envs, episode_length, 1)
  logged_time_array = logged_time_array.transpose(1, 0, 2)

  # Save to file
  if output_file is None:
    output_file = "rl_qpos_qvel.npz"

  output_path = Path(output_file)
  print(f"[INFO]: Saving logged data to {output_path}")
  # Get dimensions safely from mj_data
  mj_data = env.unwrapped.sim.data
  qpos_tensor = (
    mj_data.qpos._tensor if hasattr(mj_data.qpos, "_tensor") else mj_data.qpos
  )
  qvel_tensor = (
    mj_data.qvel._tensor if hasattr(mj_data.qvel, "_tensor") else mj_data.qvel
  )
  np.savez_compressed(
    output_path,
    x=logged_array,
    time=logged_time_array,
    num_envs=num_envs,
    episode_length=episode_length,
  )
  print(f"[INFO]: Saved data shape: {logged_array.shape}")
  print(f"[INFO]: Data saved successfully to {output_path.absolute()}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  @dataclass(frozen=True)
  class LogConfig(PlayConfig):
    output_file: str | None = None

  chosen_task, args = parse_choice_and_dataclass(
    all_tasks,
    LogConfig,
    prog=sys.argv[0],
    description=__doc__,
  )

  run_log_qpos_qvel(chosen_task, args, args.output_file)


if __name__ == "__main__":
  main()
