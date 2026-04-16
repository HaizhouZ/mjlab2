"""Tests specific to motion tracking tasks."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from mjlab.asset_zoo.robots import G1_ACTION_SCALE
from mjlab.envs.mdp.actions import JointPositionActionCfg
from mjlab.scripts.train_trajectory_encoder import (
  TrajectoryEncoderModelCfg,
  build_model,
)
from mjlab.tasks.registry import list_tasks, load_env_cfg
from mjlab.tasks.tracking.mdp import MotionCommandCfg, MultiMotionCommandCfg
from mjlab.tasks.tracking.mdp import observations as tracking_observations
from mjlab.tasks.tracking.mdp.commands import (
  _load_trajectory_encoder_checkpoint,
  TrajectoryEncoderLoadInfo,
  resolve_trajectory_encoder_load_info,
  validate_trajectory_encoder_output,
)


@pytest.fixture(scope="module")
def tracking_task_ids() -> list[str]:
  """Get all tracking task IDs."""
  return [t for t in list_tasks() if "Tracking" in t]


@pytest.fixture(scope="module")
def g1_tracking_task_ids(tracking_task_ids: list[str]) -> list[str]:
  """Get all G1 tracking task IDs."""
  return [t for t in tracking_task_ids if "G1" in t]


def test_tracking_tasks_have_motion_command(tracking_task_ids: list[str]) -> None:
  """All tracking tasks should have a 'motion' command of type MotionCommandCfg."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "motion" in cfg.commands, f"Task {task_id} missing 'motion' command"

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, (MotionCommandCfg, MultiMotionCommandCfg)), (
      f"Task {task_id} motion command has unexpected type {type(motion_cmd)}"
    )


def test_tracking_tasks_have_self_collision_sensor(
  tracking_task_ids: list[str],
) -> None:
  """All tracking tasks should have a self_collision sensor."""
  for task_id in tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert cfg.scene.sensors is not None, f"Task {task_id} has no sensors"

    sensor_names = {s.name for s in cfg.scene.sensors}
    assert "self_collision" in sensor_names, (
      f"Task {task_id} missing self_collision sensor"
    )


def test_tracking_no_state_estimation_observations() -> None:
  """No-state-estimation tasks remove observations that depend on state estimation."""
  task_id = "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation"

  # Test both training and play modes
  for play_mode in [False, True]:
    cfg = load_env_cfg(task_id, play=play_mode)
    mode_str = "play mode" if play_mode else "training mode"

    assert "policy" in cfg.observations, (
      f"Task {task_id} ({mode_str}) missing policy observations"
    )
    policy_terms = cfg.observations["policy"].terms

    assert "motion_anchor_pos_b" not in policy_terms, (
      f"Task {task_id} ({mode_str}) has motion_anchor_pos_b in policy, "
      "expected it to be removed for no-state-estimation variant"
    )
    assert "base_lin_vel" not in policy_terms, (
      f"Task {task_id} ({mode_str}) has base_lin_vel in policy, "
      "expected it to be removed for no-state-estimation variant"
    )


def test_tracking_play_disables_rsi_randomization() -> None:
  """Tracking play tasks should disable RSI randomization."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.pose_range == {}, (
      f"Task {task_id} (play mode) has non-empty pose_range={motion_cmd.pose_range}, "
      "expected empty dict for disabled RSI"
    )
    assert motion_cmd.velocity_range == {}, (
      f"Task {task_id} (play mode) has non-empty velocity_range={motion_cmd.velocity_range}, "
      "expected empty dict for disabled RSI"
    )


def test_tracking_play_uses_start_sampling_mode() -> None:
  """Tracking play tasks should use sampling_mode='start'."""
  tracking_tasks = [
    "Mjlab-Tracking-Flat-Unitree-G1",
    "Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  ]

  for task_id in tracking_tasks:
    cfg = load_env_cfg(task_id, play=True)

    motion_cmd = cfg.commands["motion"]
    assert isinstance(motion_cmd, MotionCommandCfg), (
      f"Task {task_id} (play mode) motion command is not MotionCommandCfg"
    )

    assert motion_cmd.sampling_mode == "start", (
      f"Task {task_id} (play mode) sampling_mode={motion_cmd.sampling_mode}, expected 'start'"
    )


def test_g1_tracking_has_correct_action_scale(g1_tracking_task_ids: list[str]) -> None:
  """G1 tracking tasks should use G1_ACTION_SCALE."""
  for task_id in g1_tracking_task_ids:
    cfg = load_env_cfg(task_id)

    assert "joint_pos" in cfg.actions, f"Task {task_id} missing 'joint_pos' action"

    joint_pos_action = cfg.actions["joint_pos"]
    assert isinstance(joint_pos_action, JointPositionActionCfg), (
      f"Task {task_id} joint_pos action is not JointPositionActionCfg"
    )

    assert joint_pos_action.scale == G1_ACTION_SCALE, (
      f"Task {task_id} action scale mismatch, expected G1_ACTION_SCALE"
    )


def test_motion_anchor_pos_b_uses_multimotion_horizon(monkeypatch: pytest.MonkeyPatch) -> None:
  """Multi-motion observation helpers should honor cfg.horizon."""

  class FakeMultiMotionCommand:
    def __init__(self) -> None:
      self.cfg = SimpleNamespace(horizon=3)
      self.robot_anchor_pos_w = torch.zeros((2, 3), dtype=torch.float32)
      self.robot_anchor_quat_w = torch.tensor(
        [[1.0, 0.0, 0.0, 0.0], [1.0, 0.0, 0.0, 0.0]], dtype=torch.float32
      )
      self.anchor_pos_w = torch.zeros((2, 3), dtype=torch.float32)
      self.anchor_quat_w = self.robot_anchor_quat_w
      self.called_horizon: int | None = None

    def get_anchor_pos_w_horizon(self, horizon: int) -> torch.Tensor:
      self.called_horizon = horizon
      return torch.zeros((2, horizon, 3), dtype=torch.float32)

    def get_anchor_quat_w_horizon(self, horizon: int) -> torch.Tensor:
      return torch.tensor(
        [[[1.0, 0.0, 0.0, 0.0]] * horizon] * 2,
        dtype=torch.float32,
      )

  fake_command = FakeMultiMotionCommand()
  fake_env = SimpleNamespace(
    command_manager=SimpleNamespace(get_term=lambda _name: fake_command)
  )

  monkeypatch.setattr(
    tracking_observations, "MultiMotionCommand", FakeMultiMotionCommand
  )

  obs = tracking_observations.motion_anchor_pos_b(fake_env, "motion")

  assert fake_command.called_horizon == 3
  assert obs.shape == (2, 9)


def test_resolve_trajectory_encoder_load_info_infers_horizon_from_config(
  tmp_path: Path,
) -> None:
  encoder_path = tmp_path / "best_encoder.jit"
  encoder_path.write_text("jit", encoding="utf-8")
  (tmp_path / "config.yaml").write_text(
    "horizon: 16\ninput_features: tracking\nmodel:\n  latent_dim: 128\n",
    encoding="utf-8",
  )

  info = resolve_trajectory_encoder_load_info(encoder_path, configured_horizon=1)

  assert info.path == encoder_path
  assert info.inferred_horizon == 16
  assert info.horizon == 16
  assert info.feature_layout == "tracking"
  assert info.latent_dim == 128


def test_resolve_trajectory_encoder_load_info_uses_checkpoint_fallback(
  tmp_path: Path,
) -> None:
  encoder_path = tmp_path / "best_encoder.jit"
  encoder_path.write_text("jit", encoding="utf-8")
  torch.save(
    {"horizon": 24, "feature_layout": "tracking", "latent_dim": 96},
    tmp_path / "best_model.pt",
  )

  info = resolve_trajectory_encoder_load_info(encoder_path, configured_horizon=1)

  assert info.inferred_horizon == 24
  assert info.horizon == 24
  assert info.feature_layout == "tracking"
  assert info.latent_dim == 96


def test_resolve_trajectory_encoder_load_info_rejects_horizon_mismatch(
  tmp_path: Path,
) -> None:
  encoder_path = tmp_path / "best_encoder.jit"
  encoder_path.write_text("jit", encoding="utf-8")
  (tmp_path / "config.yaml").write_text(
    "horizon: 16\ninput_features: tracking\n",
    encoding="utf-8",
  )

  with pytest.raises(ValueError, match="cfg.horizon=8, encoder_horizon=16"):
    resolve_trajectory_encoder_load_info(encoder_path, configured_horizon=8)


def test_resolve_trajectory_encoder_load_info_rejects_unsupported_feature_layout(
  tmp_path: Path,
) -> None:
  encoder_path = tmp_path / "best_encoder.jit"
  encoder_path.write_text("jit", encoding="utf-8")
  (tmp_path / "config.yaml").write_text(
    "horizon: 16\ninput_features: tracking_with_object\n",
    encoding="utf-8",
  )

  with pytest.raises(ValueError, match="feature_layout='tracking_with_object'"):
    resolve_trajectory_encoder_load_info(encoder_path, configured_horizon=1)


def test_validate_trajectory_encoder_output_rejects_latent_dim_mismatch() -> None:
  class FakeEncoder(torch.nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
      return torch.zeros((x.shape[0], 32), dtype=x.dtype, device=x.device)

  load_info = TrajectoryEncoderLoadInfo(
    path=Path("/tmp/fake_encoder.jit"),
    horizon=16,
    inferred_horizon=16,
    feature_layout="tracking",
    latent_dim=64,
  )

  with pytest.raises(
    ValueError,
    match="encoder_output_dim=32, expected_latent_dim=64",
  ):
    validate_trajectory_encoder_output(
      FakeEncoder(),
      load_info,
      device=torch.device("cpu"),
    )


def test_load_trajectory_encoder_checkpoint_restores_model_from_best_model(
  tmp_path: Path,
) -> None:
  model_cfg = TrajectoryEncoderModelCfg(architecture="unet_simple", latent_dim=8)
  model = build_model(model_cfg, horizon=16, input_dim=65)
  checkpoint_path = tmp_path / "best_model.pt"
  torch.save(
    {
      "model_state_dict": model.state_dict(),
      "architecture": "unet_simple",
      "latent_dim": 8,
      "horizon": 16,
      "input_dim": 65,
      "feature_layout": "tracking",
      "config": {"model": {"architecture": "unet_simple", "latent_dim": 8}},
    },
    checkpoint_path,
  )

  load_info = resolve_trajectory_encoder_load_info(
    checkpoint_path,
    configured_horizon=1,
  )
  restored = _load_trajectory_encoder_checkpoint(
    load_info,
    device=torch.device("cpu"),
  )
  encoded = restored(torch.zeros((1, 16, 65), dtype=torch.float32))

  assert load_info.path == checkpoint_path
  assert load_info.checkpoint_path == checkpoint_path
  assert load_info.latent_dim == 8
  assert tuple(encoded.shape) == (1, 8)
