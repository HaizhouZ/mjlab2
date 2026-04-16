"""Tests for the trajectory encoder training script."""

from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import mjlab.scripts.train_trajectory_encoder as train_encoder_module
import mjlab.utils.wandb as wandb_utils_module
from mjlab.scripts.train_trajectory_encoder import (
  TrajectoryEncoderModelCfg,
  TrajectoryEncoderTrainConfig,
  TrajectoryWindowDataset,
  build_dataloader_kwargs,
  build_model,
  feature_slices,
  prepare_output_dir,
  resolve_amp_dtype,
  resolve_device,
  save_checkpoint,
  upload_wandb_artifact,
)
from mjlab.utils import parse_dataclass_cli


def test_feature_slices_match_tracking_runtime_layout() -> None:
  tracking_total = sum(slc.stop - slc.start for _, slc in feature_slices("tracking"))
  tracking_with_object_total = sum(
    slc.stop - slc.start for _, slc in feature_slices("tracking_with_object")
  )

  assert tracking_total == 65
  assert tracking_with_object_total == 72


def test_config_path_and_cli_override(tmp_path: Path) -> None:
  config_path = tmp_path / "encoder.yaml"
  config_path.write_text(
    "\n".join(
      [
        "motion_dir: motions",
        "horizon: 24",
        "model:",
        "  architecture: unet_simple",
        "  latent_dim: 64",
      ]
    ),
    encoding="utf-8",
  )

  cfg = parse_dataclass_cli(
    TrajectoryEncoderTrainConfig,
    argv=[
      f"config_path={config_path}",
      "horizon=48",
      "model.latent_dim=96",
    ],
  )

  assert cfg.motion_dir == "motions"
  assert cfg.horizon == 48
  assert cfg.model.architecture == "unet_simple"
  assert cfg.model.latent_dim == 96


def test_checkpoint_save_persists_training_metadata(tmp_path: Path) -> None:
  cfg = TrajectoryEncoderTrainConfig(
    motion_dir="motions",
    model=TrajectoryEncoderModelCfg(architecture="unet_simple", latent_dim=8),
  )
  model = build_model(cfg.model, horizon=4, input_dim=65)
  optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)

  checkpoint_path = tmp_path / "best_model.pt"
  save_checkpoint(
    checkpoint_path,
    cfg=cfg,
    epoch=3,
    model=model,
    optimizer=optimizer,
    train_loss=0.25,
    val_loss=0.2,
    input_dim=65,
  )

  payload = torch.load(checkpoint_path, map_location="cpu")
  assert payload["epoch"] == 3
  assert payload["feature_layout"] == "tracking"
  assert payload["input_dim"] == 65
  assert payload["architecture"] == "unet_simple"
  assert "model_state_dict" in payload
  assert "optimizer_state_dict" in payload


def test_tracking_with_object_requires_object_pose_fields() -> None:
  dataset = object.__new__(TrajectoryWindowDataset)
  dataset.include_object_pose = True

  with pytest.raises(
    ValueError, match="tracking_with_object.*object_pos_w, object_quat_w"
  ):
    dataset._validate_motion_features(SimpleNamespace())


def test_resolve_device_falls_back_to_cpu_when_cuda_missing(monkeypatch) -> None:
  monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
  assert resolve_device("cuda").type == "cpu"


def test_gpu_training_defaults_enable_fast_path() -> None:
  cfg = TrajectoryEncoderTrainConfig()
  assert cfg.num_workers == 4
  assert cfg.mixed_precision is True
  assert cfg.amp_dtype == "float16"
  assert resolve_amp_dtype(cfg) == torch.float16

  kwargs = build_dataloader_kwargs(
    batch_size=32,
    shuffle=True,
    num_workers=cfg.num_workers,
    device=torch.device("cuda"),
    prefetch_factor=cfg.prefetch_factor,
  )
  assert kwargs["pin_memory"] is True
  assert kwargs["persistent_workers"] is True
  assert kwargs["prefetch_factor"] == cfg.prefetch_factor


def test_prepare_output_dir_uses_mjlab_output_dir_for_relative_logs(
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  monkeypatch.setenv("MJLAB_OUTPUT_DIR", "/tmp/mjlab-output")
  cfg = TrajectoryEncoderTrainConfig(output_dir="logs/trajectory_encoder", run_name="test")
  output_dir = prepare_output_dir(cfg)
  assert str(output_dir).startswith("/tmp/mjlab-output/trajectory_encoder/test/")


def test_upload_wandb_artifact_collects_encoder_outputs(tmp_path: Path, monkeypatch) -> None:
  class FakeArtifact:
    def __init__(self, name: str, type: str, metadata: dict):
      self.name = name
      self.type = type
      self.metadata = metadata
      self.files: list[tuple[str, str]] = []

    def add_file(self, path: str, name: str | None = None) -> None:
      self.files.append((path, name or Path(path).name))

  class FakeRun:
    def __init__(self) -> None:
      self.id = "run123"
      self.logged: tuple[FakeArtifact, list[str]] | None = None

    def log_artifact(self, artifact: FakeArtifact, aliases: list[str]):
      self.logged = (artifact, aliases)
      return artifact

  fake_wandb = SimpleNamespace(Artifact=FakeArtifact)
  monkeypatch.setattr(train_encoder_module, "wandb", fake_wandb)

  cfg = TrajectoryEncoderTrainConfig(
    wandb_log=train_encoder_module.WandbLoggingCfg(
      enabled=True,
      project="traj_encoder",
      run_name="traj_run",
      upload_artifact=True,
    )
  )
  for filename in [
    "best_model.pt",
    "last_model.pt",
    "config.yaml",
    "normalizer.pt",
  ]:
    (tmp_path / filename).write_text("x", encoding="utf-8")

  run = FakeRun()
  artifact = upload_wandb_artifact(
    run,
    cfg=cfg,
    output_dir=tmp_path,
    best_epoch=4,
    best_val_loss=0.123,
  )

  assert artifact is not None
  assert run.logged is not None
  logged_artifact, aliases = run.logged
  assert logged_artifact.name == "traj_run"
  assert logged_artifact.type == "trajectory_encoder"
  assert logged_artifact.metadata["horizon"] == cfg.horizon
  assert logged_artifact.metadata["feature_layout"] == cfg.input_features
  assert "latest" in aliases
  assert "best" in aliases
  assert "run123" in aliases
  uploaded_names = [name for _, name in logged_artifact.files]
  assert "best_model.pt" in uploaded_names
  assert "best_encoder.jit" not in uploaded_names


def test_resolve_wandb_file_path_uses_mjlab_cache_root(
  tmp_path: Path,
  monkeypatch: pytest.MonkeyPatch,
) -> None:
  class FakeArtifact:
    def __init__(self) -> None:
      self.root: str | None = None

    def download(self, root: str | None = None) -> str:
      self.root = root
      local_dir = Path(root or tmp_path) / "entity" / "project" / "artifact"
      local_dir.mkdir(parents=True, exist_ok=True)
      (local_dir / "best_model.pt").write_text("x", encoding="utf-8")
      return str(local_dir)

  class FakeApi:
    def __init__(self, artifact: FakeArtifact) -> None:
      self._artifact = artifact

    def artifact(self, _name: str) -> FakeArtifact:
      return self._artifact

  fake_artifact = FakeArtifact()
  monkeypatch.setenv("MJLAB_WANDB_CACHE_DIR", str(tmp_path / "wandb-cache"))
  monkeypatch.setattr(
    wandb_utils_module,
    "wandb",
    SimpleNamespace(Api=lambda: FakeApi(fake_artifact)),
  )

  resolved = wandb_utils_module.resolve_wandb_file_path(
    "wandb://entity/project/artifact:latest/best_model.pt"
  )

  assert resolved.name == "best_model.pt"
  assert fake_artifact.root == str(tmp_path / "wandb-cache")
