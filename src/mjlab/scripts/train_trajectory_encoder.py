"""Train a trajectory encoder for tracking tasks."""

import contextlib
import os
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset, Subset

try:
  import wandb  # type: ignore

  WANDB_AVAILABLE = True
except ImportError:
  WANDB_AVAILABLE = False
  wandb = None  # type: ignore

from mjlab.asset_zoo.robots import (
  get_g1_robot_cfg,
  get_go1_robot_cfg,
  get_yam_robot_cfg,
)
from mjlab.entity import Entity
from mjlab.models import (
  TrajectoryAutoencoder2DCNN,
  TrajectoryAutoencoder2DCNNCausal,
  TrajectoryAutoencoderBase,
  TrajectoryAutoencoderFSQ,
  TrajectoryAutoencoderTCN,
  TrajectoryAutoencoderTransformer,
  TrajectoryAutoencoderUNet,
  TrajectoryAutoencoderUNetResidual,
  TrajectoryAutoencoderUNetSimple,
  TrajectoryAutoencoderUNetSimpleResidual,
  TrajectoryAutoencoderUNetSimpleTemporal,
)
from mjlab.models.normalization import TrajectoryNormalizer
from mjlab.models.trajectory_autoencoder_2dcnn_transformer import (
  TrajectoryAutoencoder2DCNNTransformer,
)
from mjlab.tasks.tracking.mdp.commands import MultiMotionLoader
from mjlab.utils import parse_dataclass_cli, save_dataclass_to_yaml
from mjlab.utils.wandb import (
  download_motions_from_wandb,
  get_wandb_motion_cache_dir,
  resolve_wandb_artifact_path,
)

DEFAULT_G1_TRACKING_BODY_NAMES = [
  "pelvis",
  "left_hip_roll_link",
  "left_knee_link",
  "left_ankle_roll_link",
  "right_hip_roll_link",
  "right_knee_link",
  "right_ankle_roll_link",
  "torso_link",
  "left_shoulder_roll_link",
  "left_elbow_link",
  "left_wrist_yaw_link",
  "right_shoulder_roll_link",
  "right_elbow_link",
  "right_wrist_yaw_link",
]

ArchitectureName = Literal[
  "unet",
  "unet_residual",
  "unet_simple",
  "unet_simple_residual",
  "unet_simple_temporal",
  "tcn",
  "2dcnn",
  "2dcnn_causal",
  "2dcnn_transformer",
  "transformer",
  "fsq",
]

FeatureLayoutName = Literal["tracking", "tracking_with_object"]
RobotName = Literal["g1", "go1", "yam"]
SchedulerName = Literal["none", "reduce_on_plateau", "cosine_annealing"]
AmpDtypeName = Literal["float16", "bfloat16"]


@dataclass(kw_only=True)
class WandbMotionSourceCfg:
  entity: str | None = None
  project: str | None = None
  cache_dir: str | None = None
  artifact_type: str = "motions"


@dataclass(kw_only=True)
class WandbLoggingCfg:
  enabled: bool = False
  entity: str | None = None
  project: str = "trajectory_encoder"
  run_name: str | None = None
  upload_artifact: bool = True
  artifact_name: str | None = None
  artifact_type: str = "trajectory_encoder"


@dataclass(kw_only=True)
class EarlyStoppingCfg:
  enabled: bool = True
  patience: int = 25
  min_delta: float = 0.0


@dataclass(kw_only=True)
class TrajectoryEncoderModelCfg:
  architecture: ArchitectureName = "unet_simple"
  latent_dim: int = 128

  encoder_channels: list[int] | None = None
  decoder_channels: list[int] | None = None
  encoder_hidden_dims: list[int] | None = None
  decoder_hidden_dims: list[int] | None = None
  num_channels: list[int] | None = None
  cnn_channels: list[int] | None = None

  use_batch_norm: bool = True
  dropout: float = 0.0
  kernel_size: int = 3
  use_residual: bool = True
  use_temporal_pos_encoding: bool = True
  use_temporal_conv: bool = False
  temporal_conv_dim: int | None = None
  temporal_conv_kernel_size: int = 3

  d_model: int = 256
  nhead: int = 8
  num_encoder_layers: int = 4
  num_decoder_layers: int = 4
  dim_feedforward: int = 1024
  activation: str = "relu"
  use_residual_connection: bool = True

  hidden_dim: int = 256
  fsq_dim: int = 8
  fsq_levels: list[int] | None = None
  num_temporal_layers: int = 2


@dataclass(kw_only=True)
class TrajectoryEncoderTrainConfig:
  motion_dir: str | None = None
  motion_name_pattern: list[str] = field(default_factory=lambda: [".*"])
  wandb_motion: WandbMotionSourceCfg = field(default_factory=WandbMotionSourceCfg)

  output_dir: str = "logs/trajectory_encoder"
  run_name: str | None = None

  robot: RobotName = "g1"
  body_names: list[str] = field(
    default_factory=lambda: list(DEFAULT_G1_TRACKING_BODY_NAMES)
  )
  anchor_body_name: str = "torso_link"

  input_features: FeatureLayoutName = "tracking"
  horizon: int = 32
  stride: int = 1

  batch_size: int = 512
  epochs: int = 200
  lr: float = 1e-3
  weight_decay: float = 0.0
  val_split: float = 0.1
  scheduler: SchedulerName = "reduce_on_plateau"
  min_lr: float | None = None
  scheduler_patience: int = 5
  scheduler_factor: float = 0.5

  device: str = "cuda"
  seed: int = 0
  num_workers: int = 4
  mixed_precision: bool = True
  amp_dtype: AmpDtypeName = "float16"
  tf32: bool = True
  cudnn_benchmark: bool = True
  compile_model: bool = False
  prefetch_factor: int = 4
  max_normalizer_samples: int = 10000
  checkpoint_every: int = 50
  save_jit: bool = True
  normalize: bool = True

  model: TrajectoryEncoderModelCfg = field(default_factory=TrajectoryEncoderModelCfg)
  early_stopping: EarlyStoppingCfg = field(default_factory=EarlyStoppingCfg)
  wandb_log: WandbLoggingCfg = field(default_factory=WandbLoggingCfg)

  config_path: str | None = None
  """Optional path to YAML/JSON config file to load defaults from."""
  config_source: str | None = None
  """Optional config source: local path, http(s) URL, or wandb:// URI."""


class JittedEncoder(nn.Module):
  """Wrapper that exposes only the encoder path for export."""

  def __init__(self, model: TrajectoryAutoencoderBase):
    super().__init__()
    self.model = model

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.model.encode(x)


class TrajectoryWindowDataset(Dataset[torch.Tensor]):
  """Windowed motion dataset for trajectory encoder training."""

  def __init__(
    self,
    *,
    motion_dir: str,
    motion_name_pattern: list[str],
    body_indexes: torch.Tensor,
    anchor_body_index: int,
    horizon: int,
    stride: int = 1,
    device: str = "cpu",
    include_object_pose: bool = False,
  ):
    self.horizon = horizon
    self.include_object_pose = include_object_pose
    self.motion_loader = MultiMotionLoader(
      motion_dir=motion_dir,
      motion_name_pattern=motion_name_pattern,
      body_indexes=body_indexes,
      device=device,
    )
    self.anchor_body_index = anchor_body_index
    self.windows: list[tuple[int, int]] = []

    for motion_idx in range(self.motion_loader.num_motions):
      motion = self.motion_loader.motions[motion_idx]
      self._validate_motion_features(motion)
      for start_t in range(0, motion.time_step_total - horizon + 1, stride):
        self.windows.append((motion_idx, start_t))

    if not self.windows:
      max_steps = max(m.time_step_total for m in self.motion_loader.motions)
      raise ValueError(
        f"No valid trajectory windows found for horizon={horizon}. "
        f"Longest motion has {max_steps} timesteps."
      )

  @property
  def input_dim(self) -> int:
    return 72 if self.include_object_pose else 65

  def _validate_motion_features(self, motion: Any) -> None:
    """Validate that motions expose the fields required by the selected layout."""
    if not self.include_object_pose:
      return

    missing = [
      name
      for name in ("object_pos_w", "object_quat_w")
      if not hasattr(motion, name)
    ]
    if missing:
      raise ValueError(
        "input_features='tracking_with_object' requires motion data with "
        f"{', '.join(missing)} fields, but they are missing from the loaded "
        "artifact motions. Use input_features='tracking' for datasets without "
        "object pose annotations."
      )

  def __len__(self) -> int:
    return len(self.windows)

  def __getitem__(self, idx: int) -> torch.Tensor:
    motion_idx, start_t = self.windows[idx]
    motion = self.motion_loader.motions[motion_idx]
    end_t = start_t + self.horizon

    joint_pos = motion.joint_pos[start_t:end_t].clone()
    joint_vel = motion.joint_vel[start_t:end_t].clone()
    body_pos_w = motion.body_pos_w[start_t:end_t].clone()
    body_quat_w = motion.body_quat_w[start_t:end_t].clone()

    features = [
      joint_pos,
      joint_vel,
      body_pos_w[:, self.anchor_body_index],
      body_quat_w[:, self.anchor_body_index],
    ]

    if self.include_object_pose:
      features.append(motion.object_pos_w[start_t:end_t].clone())
      features.append(motion.object_quat_w[start_t:end_t].clone())

    return torch.cat(features, dim=1)


def feature_slices(layout: FeatureLayoutName) -> list[tuple[str, slice]]:
  slices = [
    ("joint_pos", slice(0, 29)),
    ("joint_vel", slice(29, 58)),
    ("anchor_pos", slice(58, 61)),
    ("anchor_quat", slice(61, 65)),
  ]
  if layout == "tracking_with_object":
    slices.extend(
      [
        ("object_pos", slice(65, 68)),
        ("object_quat", slice(68, 72)),
      ]
    )
  return slices


def compute_reconstruction_metrics(
  original: torch.Tensor,
  reconstructed: torch.Tensor,
  layout: FeatureLayoutName,
) -> dict[str, float]:
  mse = (original - reconstructed).pow(2)
  mae = (original - reconstructed).abs()
  metrics = {
    "reconstruction_mse": mse.mean().item(),
    "reconstruction_mae": mae.mean().item(),
  }
  for name, slc in feature_slices(layout):
    metrics[f"{name}_mse"] = mse[:, :, slc].mean().item()
    metrics[f"{name}_mae"] = mae[:, :, slc].mean().item()
  return metrics


def train_epoch(
  model: TrajectoryAutoencoderBase,
  dataloader: DataLoader,
  optimizer: torch.optim.Optimizer,
  criterion: nn.Module,
  *,
  device: torch.device,
  layout: FeatureLayoutName,
  use_amp: bool,
  amp_dtype: torch.dtype | None,
  scaler: torch.amp.GradScaler | None,
) -> tuple[float, dict[str, float]]:
  model.train()
  total_loss = 0.0
  total_metrics = {name: 0.0 for name in compute_metric_keys(layout)}

  for batch in dataloader:
    batch = batch.to(device, non_blocking=device.type == "cuda")
    optimizer.zero_grad(set_to_none=True)

    autocast_context = (
      torch.autocast("cuda", dtype=amp_dtype)
      if use_amp and amp_dtype is not None
      else contextlib.nullcontext()
    )
    with autocast_context:
      batch_normalized = (
        model.normalizer(batch)
        if getattr(model, "normalizer", None) is not None
        else batch
      )
      reconstructed, _ = model(batch)
      loss = criterion(reconstructed, batch_normalized)

    if scaler is not None:
      scaler.scale(loss).backward()
      scaler.step(optimizer)
      scaler.update()
    else:
      loss.backward()
      optimizer.step()

    with torch.no_grad():
      metrics = compute_reconstruction_metrics(
        batch_normalized.detach().float(),
        reconstructed.detach().float(),
        layout,
      )
    total_loss += loss.item()
    for key, value in metrics.items():
      total_metrics[key] += value

  num_batches = max(len(dataloader), 1)
  for key in total_metrics:
    total_metrics[key] /= num_batches
  return total_loss / num_batches, total_metrics


def evaluate(
  model: TrajectoryAutoencoderBase,
  dataloader: DataLoader,
  criterion: nn.Module,
  *,
  device: torch.device,
  layout: FeatureLayoutName,
  use_amp: bool,
  amp_dtype: torch.dtype | None,
) -> tuple[float, dict[str, float], float]:
  model.eval()
  total_loss = 0.0
  total_metrics = {name: 0.0 for name in compute_metric_keys(layout)}

  if device.type == "cuda":
    torch.cuda.synchronize(device)
  start = time.time()

  with torch.no_grad():
    for batch in dataloader:
      batch = batch.to(device, non_blocking=device.type == "cuda")
      autocast_context = (
        torch.autocast("cuda", dtype=amp_dtype)
        if use_amp and amp_dtype is not None
        else contextlib.nullcontext()
      )
      with autocast_context:
        batch_normalized = (
          model.normalizer(batch)
          if getattr(model, "normalizer", None) is not None
          else batch
        )
        reconstructed, _ = model(batch)
        loss = criterion(reconstructed, batch_normalized)

      metrics = compute_reconstruction_metrics(
        batch_normalized.detach().float(),
        reconstructed.detach().float(),
        layout,
      )
      total_loss += loss.item()
      for key, value in metrics.items():
        total_metrics[key] += value

  if device.type == "cuda":
    torch.cuda.synchronize(device)
  elapsed = time.time() - start

  num_batches = max(len(dataloader), 1)
  for key in total_metrics:
    total_metrics[key] /= num_batches
  return total_loss / num_batches, total_metrics, elapsed


def compute_metric_keys(layout: FeatureLayoutName) -> list[str]:
  keys = ["reconstruction_mse", "reconstruction_mae"]
  for name, _ in feature_slices(layout):
    keys.append(f"{name}_mse")
    keys.append(f"{name}_mae")
  return keys


def fit_normalizer(
  dataset: Dataset[torch.Tensor],
  indices: list[int],
  *,
  device: torch.device,
  num_workers: int,
  max_samples: int,
  batch_size: int,
) -> TrajectoryNormalizer:
  sample_indices = indices[: min(len(indices), max_samples)]
  subset = Subset(dataset, sample_indices)
  loader = DataLoader(
    subset,
    **build_dataloader_kwargs(
      batch_size=min(batch_size, max(len(sample_indices), 1)),
      shuffle=False,
      num_workers=num_workers,
      device=device,
      prefetch_factor=2,
    ),
  )

  feature_sum: torch.Tensor | None = None
  feature_sq_sum: torch.Tensor | None = None
  total_count = 0

  for batch in loader:
    batch = batch.to(device=device, dtype=torch.float32)
    flat = batch.reshape(-1, batch.shape[-1]).double()
    batch_sum = flat.sum(dim=0)
    batch_sq_sum = flat.square().sum(dim=0)
    feature_sum = batch_sum if feature_sum is None else feature_sum + batch_sum
    feature_sq_sum = (
      batch_sq_sum if feature_sq_sum is None else feature_sq_sum + batch_sq_sum
    )
    total_count += flat.shape[0]

  if feature_sum is None or feature_sq_sum is None or total_count == 0:
    raise ValueError("Unable to fit normalizer from an empty dataset.")

  mean = feature_sum / total_count
  variance = (feature_sq_sum / total_count) - mean.square()
  std = variance.clamp_min(1e-8).sqrt()
  return TrajectoryNormalizer(mean=mean.float(), std=std.float())


def resolve_amp_dtype(cfg: TrajectoryEncoderTrainConfig) -> torch.dtype | None:
  if not cfg.mixed_precision:
    return None
  if cfg.amp_dtype == "float16":
    return torch.float16
  if cfg.amp_dtype == "bfloat16":
    return torch.bfloat16
  raise ValueError(f"Unsupported amp_dtype: {cfg.amp_dtype}")


def configure_gpu_runtime(cfg: TrajectoryEncoderTrainConfig, device: torch.device) -> None:
  if device.type != "cuda":
    return

  torch.backends.cuda.matmul.allow_tf32 = cfg.tf32
  torch.backends.cudnn.allow_tf32 = cfg.tf32
  torch.backends.cudnn.benchmark = cfg.cudnn_benchmark
  torch.set_float32_matmul_precision("high" if cfg.tf32 else "highest")


def build_dataloader_kwargs(
  *,
  batch_size: int,
  shuffle: bool,
  num_workers: int,
  device: torch.device,
  prefetch_factor: int,
) -> dict[str, Any]:
  kwargs: dict[str, Any] = {
    "batch_size": batch_size,
    "shuffle": shuffle,
    "num_workers": num_workers,
    "pin_memory": device.type == "cuda",
  }
  if num_workers > 0:
    kwargs["persistent_workers"] = True
    kwargs["prefetch_factor"] = prefetch_factor
  return kwargs


def build_model(
  cfg: TrajectoryEncoderModelCfg,
  *,
  horizon: int,
  input_dim: int,
) -> TrajectoryAutoencoderBase:
  if cfg.architecture == "unet":
    return TrajectoryAutoencoderUNet(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=cfg.latent_dim,
      encoder_channels=cfg.encoder_channels,
      decoder_channels=cfg.decoder_channels,
      use_batch_norm=cfg.use_batch_norm,
    )
  if cfg.architecture == "unet_residual":
    return TrajectoryAutoencoderUNetResidual(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=cfg.latent_dim,
      encoder_channels=cfg.encoder_channels,
      decoder_channels=cfg.decoder_channels,
      use_batch_norm=cfg.use_batch_norm,
    )
  if cfg.architecture == "unet_simple":
    return TrajectoryAutoencoderUNetSimple(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=cfg.latent_dim,
      encoder_hidden_dims=cfg.encoder_hidden_dims,
      decoder_hidden_dims=cfg.decoder_hidden_dims,
      use_batch_norm=cfg.use_batch_norm,
      dropout=cfg.dropout,
    )
  if cfg.architecture == "unet_simple_residual":
    return TrajectoryAutoencoderUNetSimpleResidual(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=cfg.latent_dim,
      encoder_hidden_dims=cfg.encoder_hidden_dims,
      decoder_hidden_dims=cfg.decoder_hidden_dims,
      use_batch_norm=cfg.use_batch_norm,
      dropout=cfg.dropout,
      use_residual=cfg.use_residual,
    )
  if cfg.architecture == "unet_simple_temporal":
    return TrajectoryAutoencoderUNetSimpleTemporal(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=cfg.latent_dim,
      encoder_hidden_dims=cfg.encoder_hidden_dims,
      decoder_hidden_dims=cfg.decoder_hidden_dims,
      use_batch_norm=cfg.use_batch_norm,
      dropout=cfg.dropout,
      use_temporal_pos_encoding=cfg.use_temporal_pos_encoding,
      use_temporal_conv=cfg.use_temporal_conv,
      temporal_conv_dim=cfg.temporal_conv_dim,
      temporal_conv_kernel_size=cfg.temporal_conv_kernel_size,
    )
  if cfg.architecture == "tcn":
    return TrajectoryAutoencoderTCN(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=cfg.latent_dim,
      num_channels=cfg.num_channels,
      kernel_size=cfg.kernel_size,
      dropout=cfg.dropout,
    )
  if cfg.architecture == "2dcnn":
    return TrajectoryAutoencoder2DCNN(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=cfg.latent_dim,
      encoder_channels=cfg.encoder_channels,
      decoder_channels=cfg.decoder_channels,
      kernel_size=cfg.kernel_size,
      use_batch_norm=cfg.use_batch_norm,
    )
  if cfg.architecture == "2dcnn_causal":
    return TrajectoryAutoencoder2DCNNCausal(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=cfg.latent_dim,
      encoder_channels=cfg.encoder_channels,
      decoder_channels=cfg.decoder_channels,
      kernel_size=cfg.kernel_size,
      use_batch_norm=cfg.use_batch_norm,
    )
  if cfg.architecture == "2dcnn_transformer":
    return TrajectoryAutoencoder2DCNNTransformer(
      horizon=horizon,
      feature_dim=input_dim,
      latent_dim=cfg.latent_dim,
      cnn_channels=cfg.cnn_channels,
      d_model=cfg.d_model,
      nhead=cfg.nhead,
      num_encoder_layers=cfg.num_encoder_layers,
      num_decoder_layers=cfg.num_decoder_layers,
      dim_feedforward=cfg.dim_feedforward,
      dropout=cfg.dropout,
      activation=cfg.activation,
      use_residual_connection=cfg.use_residual_connection,
    )
  if cfg.architecture == "transformer":
    return TrajectoryAutoencoderTransformer(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=cfg.latent_dim,
      d_model=cfg.d_model,
      nhead=cfg.nhead,
      num_encoder_layers=cfg.num_encoder_layers,
      num_decoder_layers=cfg.num_decoder_layers,
      dim_feedforward=cfg.dim_feedforward,
      dropout=cfg.dropout,
      activation=cfg.activation,
    )
  if cfg.architecture == "fsq":
    return TrajectoryAutoencoderFSQ(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=cfg.latent_dim,
      hidden_dim=cfg.hidden_dim,
      fsq_dim=cfg.fsq_dim,
      fsq_levels=cfg.fsq_levels,
      num_temporal_layers=cfg.num_temporal_layers,
      use_batch_norm=cfg.use_batch_norm,
      dropout=cfg.dropout,
    )
  raise ValueError(f"Unsupported architecture: {cfg.architecture}")


def build_checkpoint_dict(
  *,
  cfg: TrajectoryEncoderTrainConfig,
  epoch: int,
  model: TrajectoryAutoencoderBase,
  optimizer: torch.optim.Optimizer,
  train_loss: float,
  val_loss: float,
  input_dim: int,
) -> dict[str, Any]:
  return {
    "epoch": epoch,
    "config": asdict(cfg),
    "model_state_dict": model.state_dict(),
    "optimizer_state_dict": optimizer.state_dict(),
    "train_loss": train_loss,
    "val_loss": val_loss,
    "architecture": cfg.model.architecture,
    "latent_dim": cfg.model.latent_dim,
    "horizon": cfg.horizon,
    "input_dim": input_dim,
    "feature_layout": cfg.input_features,
    "normalizer_state_dict": (
      model.normalizer.state_dict()
      if getattr(model, "normalizer", None) is not None
      else None
    ),
  }


def save_checkpoint(
  output_path: Path,
  *,
  cfg: TrajectoryEncoderTrainConfig,
  epoch: int,
  model: TrajectoryAutoencoderBase,
  optimizer: torch.optim.Optimizer,
  train_loss: float,
  val_loss: float,
  input_dim: int,
) -> None:
  output_path.parent.mkdir(parents=True, exist_ok=True)
  torch.save(
    build_checkpoint_dict(
      cfg=cfg,
      epoch=epoch,
      model=model,
      optimizer=optimizer,
      train_loss=train_loss,
      val_loss=val_loss,
      input_dim=input_dim,
    ),
    output_path,
  )


def save_jitted_encoder(
  model: TrajectoryAutoencoderBase,
  output_path: Path,
  *,
  device: torch.device,
) -> None:
  model.eval()
  wrapper = JittedEncoder(model).to(device)
  wrapper.eval()
  with torch.no_grad():
    scripted = torch.jit.script(wrapper)
  scripted.save(str(output_path))


def resolve_device(device_name: str) -> torch.device:
  if device_name.startswith("cuda") and not torch.cuda.is_available():
    print("CUDA not available, falling back to CPU.")
    return torch.device("cpu")
  return torch.device(device_name)


def resolve_robot_cfg(robot: RobotName):
  if robot == "g1":
    return get_g1_robot_cfg()
  if robot == "go1":
    return get_go1_robot_cfg()
  if robot == "yam":
    return get_yam_robot_cfg()
  raise ValueError(f"Unsupported robot: {robot}")


def resolve_motion_dir(cfg: TrajectoryEncoderTrainConfig) -> Path:
  if cfg.motion_dir is not None:
    return resolve_wandb_artifact_path(cfg.motion_dir)

  wandb_motion = cfg.wandb_motion
  if wandb_motion.entity is None or wandb_motion.project is None:
    raise ValueError(
      "Provide either `motion_dir` or both `wandb_motion.entity` and "
      "`wandb_motion.project`."
    )

  cache_dir = (
    Path(wandb_motion.cache_dir) if wandb_motion.cache_dir is not None else None
  )
  motion_dir = get_wandb_motion_cache_dir(
    wandb_entity=wandb_motion.entity,
    wandb_project=wandb_motion.project,
    cache_dir=cache_dir,
  )
  if not motion_dir.exists():
    motion_dir = download_motions_from_wandb(
      wandb_entity=wandb_motion.entity,
      wandb_project=wandb_motion.project,
      cache_dir=cache_dir,
      artifact_type=wandb_motion.artifact_type,
    )
  return motion_dir


def validate_config(cfg: TrajectoryEncoderTrainConfig) -> None:
  if cfg.horizon <= 0:
    raise ValueError("`horizon` must be positive.")
  if cfg.stride <= 0:
    raise ValueError("`stride` must be positive.")
  if cfg.batch_size <= 0:
    raise ValueError("`batch_size` must be positive.")
  if cfg.epochs <= 0:
    raise ValueError("`epochs` must be positive.")
  if not 0.0 < cfg.val_split < 1.0:
    raise ValueError("`val_split` must be between 0 and 1.")
  if cfg.anchor_body_name not in cfg.body_names:
    raise ValueError(
      f"`anchor_body_name`={cfg.anchor_body_name!r} must exist in `body_names`."
    )


def prepare_output_dir(cfg: TrajectoryEncoderTrainConfig) -> Path:
  output_root = Path(cfg.output_dir)
  env_output_dir = os.environ.get("MJLAB_OUTPUT_DIR")
  if not output_root.is_absolute() and env_output_dir:
    if output_root.parts[:1] == ("logs",):
      output_root = Path(*output_root.parts[1:]) if len(output_root.parts) > 1 else Path()
    output_root = Path(env_output_dir) / output_root

  run_name = cfg.run_name or (
    f"{cfg.model.architecture}_h{cfg.horizon}_l{cfg.model.latent_dim}"
  )
  timestamp = time.strftime("%Y-%m-%d/%H-%M-%S")
  output_dir = output_root / run_name / timestamp
  output_dir.mkdir(parents=True, exist_ok=True)
  return output_dir


def init_wandb(cfg: TrajectoryEncoderTrainConfig, *, output_dir: Path) -> Any | None:
  if not cfg.wandb_log.enabled:
    return None
  if not WANDB_AVAILABLE or wandb is None:
    raise ImportError("wandb logging requested but wandb is not available.")

  run_name = cfg.wandb_log.run_name or output_dir.parent.name
  return wandb.init(
    entity=cfg.wandb_log.entity,
    project=cfg.wandb_log.project,
    name=run_name,
    dir=str(output_dir),
    config=asdict(cfg),
  )


def upload_wandb_artifact(
  run: Any,
  *,
  cfg: TrajectoryEncoderTrainConfig,
  output_dir: Path,
  best_epoch: int,
  best_val_loss: float,
) -> Any | None:
  """Upload trained encoder outputs as a reusable W&B artifact."""
  if not cfg.wandb_log.upload_artifact:
    return None
  if wandb is None:
    raise ImportError("wandb artifact upload requested but wandb is not available.")

  artifact_name = (
    cfg.wandb_log.artifact_name or cfg.wandb_log.run_name or output_dir.parent.name
  )
  artifact = wandb.Artifact(
    name=artifact_name,
    type=cfg.wandb_log.artifact_type,
    metadata={
      "architecture": cfg.model.architecture,
      "latent_dim": cfg.model.latent_dim,
      "horizon": cfg.horizon,
      "feature_layout": cfg.input_features,
      "best_epoch": best_epoch + 1,
      "best_val_loss": best_val_loss,
    },
  )

  files_to_upload = [
    "best_model.pt",
    "last_model.pt",
    "config.yaml",
    "normalizer.pt",
  ]
  uploaded_files = 0
  for filename in files_to_upload:
    path = output_dir / filename
    if path.exists():
      artifact.add_file(str(path), name=filename)
      uploaded_files += 1

  if uploaded_files == 0:
    print("No encoder outputs found to upload as a W&B artifact.")
    return None

  aliases = ["latest", "best"]
  run_id = getattr(run, "id", None)
  if run_id:
    aliases.append(str(run_id))

  logged_artifact = run.log_artifact(artifact, aliases=aliases)
  print(
    "Uploaded W&B artifact "
    f"{artifact_name} ({cfg.wandb_log.artifact_type}) with aliases {aliases}"
  )
  return logged_artifact


def main() -> None:
  cfg = parse_dataclass_cli(
    TrajectoryEncoderTrainConfig,
    description=(
      "Train a trajectory encoder for tracking observations. "
      "By default the encoder is trained on the same 65-feature trajectory "
      "layout used by `mjlab.tasks.tracking.mdp.observations.trajectory_encoding`."
    ),
  )
  run_training(cfg)


def run_training(cfg: TrajectoryEncoderTrainConfig) -> None:
  validate_config(cfg)
  device = resolve_device(cfg.device)
  configure_gpu_runtime(cfg, device)
  amp_dtype = resolve_amp_dtype(cfg) if device.type == "cuda" else None
  use_amp = device.type == "cuda" and amp_dtype is not None

  random.seed(cfg.seed)
  torch.manual_seed(cfg.seed)
  if device.type == "cuda":
    torch.cuda.manual_seed_all(cfg.seed)

  output_dir = prepare_output_dir(cfg)
  save_dataclass_to_yaml(cfg, output_dir / "config.yaml")
  print(f"Output directory: {output_dir}")

  motion_dir = resolve_motion_dir(cfg)
  print(f"Motion source: {motion_dir}")

  robot = Entity(resolve_robot_cfg(cfg.robot))
  body_indexes = torch.tensor(
    robot.find_bodies(cfg.body_names, preserve_order=True)[0],
    dtype=torch.long,
    device="cpu",
  )
  anchor_body_index = cfg.body_names.index(cfg.anchor_body_name)

  dataset = TrajectoryWindowDataset(
    motion_dir=str(motion_dir),
    motion_name_pattern=cfg.motion_name_pattern,
    body_indexes=body_indexes,
    anchor_body_index=anchor_body_index,
    horizon=cfg.horizon,
    stride=cfg.stride,
    device="cpu",
    include_object_pose=(cfg.input_features == "tracking_with_object"),
  )
  print(f"Loaded {len(dataset)} trajectory windows with input_dim={dataset.input_dim}")

  val_size = max(1, int(round(len(dataset) * cfg.val_split)))
  train_size = len(dataset) - val_size
  if train_size <= 0:
    raise ValueError(
      f"Not enough data for train/val split: dataset_size={len(dataset)}, "
      f"val_split={cfg.val_split}."
    )

  split_generator = torch.Generator().manual_seed(cfg.seed)
  train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size], generator=split_generator
  )

  normalizer = None
  if cfg.normalize:
    print("Fitting normalizer from training windows...")
    train_indices = list(train_dataset.indices)
    normalizer = fit_normalizer(
      dataset,
      train_indices,
      device=device,
      num_workers=cfg.num_workers,
      max_samples=cfg.max_normalizer_samples,
      batch_size=cfg.batch_size,
    )
    torch.save(normalizer.state_dict(), output_dir / "normalizer.pt")

  train_loader = DataLoader(
    train_dataset,
    **build_dataloader_kwargs(
      batch_size=cfg.batch_size,
      shuffle=True,
      num_workers=cfg.num_workers,
      device=device,
      prefetch_factor=cfg.prefetch_factor,
    ),
  )
  val_loader = DataLoader(
    val_dataset,
    **build_dataloader_kwargs(
      batch_size=cfg.batch_size,
      shuffle=False,
      num_workers=cfg.num_workers,
      device=device,
      prefetch_factor=cfg.prefetch_factor,
    ),
  )

  base_model = build_model(
    cfg.model, horizon=cfg.horizon, input_dim=dataset.input_dim
  ).to(
    device
  )
  if normalizer is not None:
    base_model.normalizer = normalizer.to(device)
  model = base_model
  if cfg.compile_model and hasattr(torch, "compile"):
    model = torch.compile(base_model)

  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total_params = sum(p.numel() for p in model.parameters())
  print(
    f"Model: {cfg.model.architecture}, latent_dim={cfg.model.latent_dim}, "
    f"trainable_params={trainable_params:,}, total_params={total_params:,}"
  )
  if device.type == "cuda":
    print(
      f"GPU optimizations: amp={use_amp} amp_dtype={cfg.amp_dtype if use_amp else 'off'} "
      f"tf32={cfg.tf32} workers={cfg.num_workers} compile={cfg.compile_model}"
    )

  run = init_wandb(cfg, output_dir=output_dir)

  optimizer = torch.optim.AdamW(
    model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
  )
  scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
  criterion = nn.MSELoss()

  scheduler: torch.optim.lr_scheduler.LRScheduler | None = None
  plateau_scheduler: torch.optim.lr_scheduler.ReduceLROnPlateau | None = None
  min_lr = cfg.min_lr if cfg.min_lr is not None else cfg.lr * 0.01
  if cfg.scheduler == "reduce_on_plateau":
    plateau_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer,
      mode="min",
      factor=cfg.scheduler_factor,
      patience=cfg.scheduler_patience,
      min_lr=min_lr,
    )
  elif cfg.scheduler == "cosine_annealing":
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer,
      T_max=cfg.epochs,
      eta_min=min_lr,
    )

  best_val_loss = float("inf")
  epochs_without_improvement = 0
  best_epoch = -1

  for epoch in range(cfg.epochs):
    train_loss, train_metrics = train_epoch(
      model,
      train_loader,
      optimizer,
      criterion,
      device=device,
      layout=cfg.input_features,
      use_amp=use_amp,
      amp_dtype=amp_dtype,
      scaler=scaler if use_amp else None,
    )
    val_loss, val_metrics, val_time = evaluate(
      model,
      val_loader,
      criterion,
      device=device,
      layout=cfg.input_features,
      use_amp=use_amp,
      amp_dtype=amp_dtype,
    )

    if plateau_scheduler is not None:
      plateau_scheduler.step(val_loss)
    if scheduler is not None:
      scheduler.step()

    current_lr = optimizer.param_groups[0]["lr"]
    print(
      f"Epoch {epoch + 1}/{cfg.epochs}: "
      f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={current_lr:.2e}"
    )

    if run is not None and wandb is not None:
      wandb.log(
        {
          "epoch": epoch + 1,
          "train/loss": train_loss,
          "val/loss": val_loss,
          "learning_rate": current_lr,
          "val/inference_time_seconds": val_time,
          **{f"train/{k}": v for k, v in train_metrics.items()},
          **{f"val/{k}": v for k, v in val_metrics.items()},
        }
      )

    improved = val_loss < (best_val_loss - cfg.early_stopping.min_delta)
    if improved:
      best_val_loss = val_loss
      best_epoch = epoch
      epochs_without_improvement = 0

      save_checkpoint(
        output_dir / "best_model.pt",
        cfg=cfg,
        epoch=epoch,
        model=base_model,
        optimizer=optimizer,
        train_loss=train_loss,
        val_loss=val_loss,
        input_dim=dataset.input_dim,
      )
      if cfg.save_jit:
        save_jitted_encoder(
          base_model, output_dir / "best_encoder.jit", device=device
        )
      print(f"  Saved new best checkpoint at epoch {epoch + 1}")
    else:
      epochs_without_improvement += 1

    if cfg.checkpoint_every > 0 and (epoch + 1) % cfg.checkpoint_every == 0:
      save_checkpoint(
        output_dir / f"checkpoint_epoch_{epoch + 1}.pt",
        cfg=cfg,
        epoch=epoch,
        model=base_model,
        optimizer=optimizer,
        train_loss=train_loss,
        val_loss=val_loss,
        input_dim=dataset.input_dim,
      )

    if (
      cfg.early_stopping.enabled
      and epochs_without_improvement >= cfg.early_stopping.patience
    ):
      print(
        "Early stopping triggered after "
        f"{epochs_without_improvement} epochs without validation improvement."
      )
      break

  save_checkpoint(
    output_dir / "last_model.pt",
    cfg=cfg,
    epoch=epoch,
    model=base_model,
    optimizer=optimizer,
    train_loss=train_loss,
    val_loss=val_loss,
    input_dim=dataset.input_dim,
  )

  print(f"Best validation loss: {best_val_loss:.6f} at epoch {best_epoch + 1}")
  if run is not None and wandb is not None:
    upload_wandb_artifact(
      run,
      cfg=cfg,
      output_dir=output_dir,
      best_epoch=best_epoch,
      best_val_loss=best_val_loss,
    )
    wandb.finish()


if __name__ == "__main__":
  main()
