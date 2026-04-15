"""Script to play RL agent with RSL-RL."""

import os
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, Optional

import torch
from rsl_rl.runners import OnPolicyRunner

from mjlab.envs import ManagerBasedRlEnv, ManagerBasedRlEnvCfg
from mjlab.rl import RslRlOnPolicyRunnerCfg, RslRlVecEnvWrapper
from mjlab.tasks.registry import list_tasks, load_env_cfg, load_rl_cfg, load_runner_cls
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiMotionCommandCfg
from mjlab.tasks.tracking.rl.exporter import (
  attach_onnx_metadata as attach_tracking_onnx_metadata,
)
from mjlab.tasks.tracking.rl.exporter import (
  export_motion_policy_as_onnx,
)
from mjlab.tasks.velocity.rl.exporter import (
  attach_onnx_metadata as attach_velocity_onnx_metadata,
)
from mjlab.utils import parse_choice_and_dataclass
from mjlab.utils.lab_api.rl.exporter import export_policy_as_onnx
from mjlab.utils.os import get_wandb_checkpoint_path
from mjlab.utils.torch import configure_torch_backends
from mjlab.utils.wandb import resolve_wandb_artifact_path
from mjlab.utils.wrappers import VideoRecorder
from mjlab.viewer import NativeMujocoViewer, ViserPlayViewer


def _get_exportable_policy(policy):
  if hasattr(policy, "as_onnx"):
    return SimpleNamespace(actor=policy.as_onnx(False), is_recurrent=False)
  if hasattr(policy, "actor") or hasattr(policy, "student"):
    return policy
  return SimpleNamespace(
    actor=policy,
    is_recurrent=getattr(policy, "is_recurrent", False),
    memory_a=getattr(policy, "memory_a", None),
    memory_s=getattr(policy, "memory_s", None),
  )


@dataclass(frozen=True)
class PlayConfig:
  env: ManagerBasedRlEnvCfg
  agent_cfg: RslRlOnPolicyRunnerCfg
  agent: Literal["zero", "random", "trained"] = "trained"
  registry_name: str | None = None
  wandb_run_path: str | None = None
  checkpoint_file: str | None = None
  motion_file: str | None = None
  motion_dir: str | None = None
  num_envs: int | None = None
  device: str | None = None
  video: bool = False
  video_length: int = 200
  video_height: int | None = None
  video_width: int | None = None
  camera: int | str | None = None
  viewer: Literal["auto", "native", "viser"] = "auto"
  config_path: Optional[str] = None
  """Optional path to YAML config file to load defaults from."""
  config_source: Optional[str] = None
  """Optional config source: local path, http(s) URL, or wandb:// URI."""

  # Internal flag used by demo script.
  _demo_mode: bool = False

  @staticmethod
  def from_task(task_id: str) -> "PlayConfig":
    env_cfg = load_env_cfg(task_id, play=True)
    agent_cfg = load_rl_cfg(task_id)
    assert isinstance(agent_cfg, RslRlOnPolicyRunnerCfg)
    return PlayConfig(env=env_cfg, agent_cfg=agent_cfg)


def run_play(task_id: str, cfg: PlayConfig):
  configure_torch_backends()

  device = cfg.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

  env_cfg = cfg.env
  agent_cfg = cfg.agent_cfg

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

  if is_tracking_task and cfg._demo_mode:
    # Demo mode: use uniform sampling to see more diversity with num_envs > 1.
    motion_cmd = env_cfg.commands["motion"]
    assert isinstance(motion_cmd, (MotionCommandCfg, MultiMotionCommandCfg))
    motion_cmd.sampling_mode = "uniform"

  if is_tracking_task:
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
        motion_cmd.traj_name_patterns = [".*"]
    else:
      if isinstance(motion_cmd, MotionCommandCfg) and cfg.motion_file is not None:
        resolved_motion_file = str(resolve_wandb_artifact_path(cfg.motion_file))
        print(f"[INFO]: Using motion file from CLI: {resolved_motion_file}")
        motion_cmd.motion_file = resolved_motion_file
      elif isinstance(motion_cmd, MultiMotionCommandCfg) and cfg.motion_dir is not None:
        motion_dir = str(resolve_wandb_artifact_path(cfg.motion_dir))
        print(f"[INFO]: Using motion directory from CLI: {motion_dir}")
        motion_cmd.motion_dir = motion_dir
        motion_cmd.traj_name_patterns = [".*"]
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
          art = next(
            (a for a in wandb_run.used_artifacts() if a.type == "motions"), None
          )
          if art is None:
            raise RuntimeError("No motion artifact found in the run.")
          if isinstance(motion_cmd, MotionCommandCfg):
            motion_cmd.motion_file = str(Path(art.download()) / "motion.npz")
          elif isinstance(motion_cmd, MultiMotionCommandCfg):
            # motion_dir = str(Path(art.download()) / "motions")
            motion_dir = str(Path(art.download()))
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
  if cfg.video_height is not None:
    env_cfg.viewer.height = cfg.video_height
  if cfg.video_width is not None:
    env_cfg.viewer.width = cfg.video_width

  render_mode = "rgb_array" if (TRAINED_MODE and cfg.video) else None
  if cfg.video and DUMMY_MODE:
    print(
      "[WARN] Video recording with dummy agents is disabled (no checkpoint/log_dir)."
    )
  env = ManagerBasedRlEnv(cfg=env_cfg, device=device, render_mode=render_mode)

  if TRAINED_MODE and cfg.video:
    print("[INFO] Recording videos during play")
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    env = VideoRecorder(
      env,
      video_folder=log_dir / "videos" / "play",
      step_trigger=lambda step: step == 0,
      video_length=cfg.video_length,
      disable_logger=True,
    )

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
    runner_cls = load_runner_cls(task_id) or OnPolicyRunner
    runner = runner_cls(env, asdict(agent_cfg), device=device)
    runner.load(str(resume_path), map_location=device)
    policy = runner.get_inference_policy(device=device)

    # Export policy as ONNX before running play
    assert log_dir is not None  # log_dir is set in TRAINED_MODE block
    onnx_path = str(log_dir)
    onnx_filename = "policy.onnx"

    # Get normalizer if it exists
    if getattr(runner.alg.policy, "actor_obs_normalization", False):
      normalizer = getattr(runner.alg.policy, "actor_obs_normalizer", None)
    elif getattr(runner.alg.policy, "obs_normalization", False):
      normalizer = getattr(runner.alg.policy, "obs_normalizer", None)
    else:
      normalizer = None

    print(f"[INFO]: Exporting policy as ONNX to {onnx_path}/{onnx_filename}")
    if is_tracking_task:
      export_motion_policy_as_onnx(
        env.unwrapped,
        _get_exportable_policy(runner.alg.policy),
        normalizer=normalizer,
        path=onnx_path,
        filename=onnx_filename,
      )
    else:
      export_policy_as_onnx(
        runner.alg.policy,
        normalizer=normalizer,
        path=onnx_path,
        filename=onnx_filename,
      )
    print("[INFO]: ONNX export completed")

    # Attach ONNX metadata
    # Use wandb_run_path if available, otherwise use log_dir name
    if cfg.wandb_run_path is not None:
      run_path = cfg.wandb_run_path
    else:
      # Use the log_dir name as fallback
      run_path = str(log_dir.name) if log_dir else "unknown"

    print(f"[INFO]: Attaching ONNX metadata (run_path: {run_path})")
    if is_tracking_task:
      attach_tracking_onnx_metadata(
        env.unwrapped,
        run_path,
        path=onnx_path,
        filename=onnx_filename,
      )
    else:
      attach_velocity_onnx_metadata(
        env.unwrapped,
        run_path,
        path=onnx_path,
        filename=onnx_filename,
      )
    print("[INFO]: ONNX metadata attachment completed")

  # Handle "auto" viewer selection.
  if cfg.viewer == "auto":
    has_display = bool(os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"))
    resolved_viewer = "native" if has_display else "viser"
    del has_display
  else:
    resolved_viewer = cfg.viewer

  if resolved_viewer == "native":
    NativeMujocoViewer(env, policy).run()
  elif resolved_viewer == "viser":
    ViserPlayViewer(env, policy).run()
  else:
    raise RuntimeError(f"Unsupported viewer backend: {resolved_viewer}")

  env.close()


def main():
  # Parse first argument to choose the task.
  # Import tasks to populate the registry.
  import mjlab.tasks  # noqa: F401

  all_tasks = list_tasks()
  chosen_task, args = parse_choice_and_dataclass(
    all_tasks,
    PlayConfig,
    default_factory=PlayConfig.from_task,
    prog=sys.argv[0],
    description=__doc__,
  )

  run_play(chosen_task, args)


if __name__ == "__main__":
  main()
