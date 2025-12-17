"""Training script for trajectory autoencoder using Hydra for configuration management."""

from __future__ import annotations

import math
import time
from pathlib import Path
from typing import Any

import hydra
import torch
import torch.nn as nn
from hydra.core.hydra_config import HydraConfig
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

try:
  import wandb  # type: ignore

  WANDB_AVAILABLE = True
except ImportError:
  WANDB_AVAILABLE = False
  wandb = None  # type: ignore

from mjlab.asset_zoo.robots import get_g1_robot_cfg
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
from mjlab.models.trajectory_dataset import TrajectoryDataset
from mjlab.utils.wandb import download_motions_from_wandb


class JittedEncoder(nn.Module):
  """Wrapper for encoder that can be JIT compiled.

  The normalizer is now part of the model, so encode() handles normalization internally.
  """

  def __init__(self, model: TrajectoryAutoencoderBase):
    """Initialize the jitted encoder wrapper.

    Args:
        model: The full autoencoder model (normalizer is attached if present)
    """
    super().__init__()
    self.model = model

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim)

    Returns:
        Latent vector of shape (batch_size, latent_dim)
    """
    # Model.encode() handles normalization internally if normalizer is attached
    return self.model.encode(x)


def save_jitted_encoder(
  model: TrajectoryAutoencoderBase,
  output_path: Path,
  device: str = "cpu",
) -> None:
  """Save a JIT-compiled encoder version of the model.

  Args:
      model: The trained model (with normalizer attached if present)
      output_path: Path to save the JIT-compiled encoder
      device: Device to use for JIT compilation
  """
  model.eval()

  # Create encoder wrapper
  encoder_wrapper = JittedEncoder(model).to(device)
  encoder_wrapper.eval()

  # Create JIT-compiled version
  try:
    with torch.no_grad():
      jitted_encoder = torch.jit.script(encoder_wrapper)

    # Save JIT-compiled encoder
    jitted_encoder.save(str(output_path))
    print(f"  ✓ JIT-compiled encoder saved to {output_path}")
  except Exception as e:
    print(f"  ⚠ Warning: Failed to save JIT-compiled encoder: {e}")
    print("    This is non-fatal - regular checkpoint was saved successfully")


def print_model_architecture(
  model: TrajectoryAutoencoderBase, input_shape: tuple[int, int, int]
) -> None:
  """Print detailed model architecture with layer sizes.

  Args:
      model: The model to print
      input_shape: Input shape (batch_size, horizon, input_dim) for testing
  """
  print("\n" + "=" * 80)
  print("Model Architecture")
  print("=" * 80)

  # Try to use torchinfo if available for better formatting
  try:
    from torchinfo import summary

    summary(model, input_size=input_shape, device="cpu", verbose=0)
  except ImportError:
    # Fallback: custom detailed architecture printing
    print(f"\nModel: {model.__class__.__name__}")
    print(f"Input shape: {input_shape}")

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Non-trainable parameters: {total_params - trainable_params:,}")

    # Calculate model size (assuming float32)
    model_size_mb = total_params * 4 / (1024 * 1024)
    print(f"Model size: ~{model_size_mb:.2f} MB")

    print("\n" + "-" * 80)
    print(f"{'Layer Name':<50} {'Layer Type & Shape':<45} {'Parameters':<15}")
    print("-" * 80)

    for name, module in model.named_modules():
      if len(list(module.children())) == 0:  # Leaf node (actual layers)
        num_params = sum(p.numel() for p in module.parameters())
        if num_params > 0 or isinstance(
          module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm, nn.Dropout)
        ):
          layer_type = type(module).__name__

          # Build layer info string
          layer_info_parts = []
          if hasattr(module, "in_features") and hasattr(module, "out_features"):
            layer_info_parts.append(f"{module.in_features}→{module.out_features}")
          elif hasattr(module, "in_channels") and hasattr(module, "out_channels"):
            layer_info_parts.append(f"{module.in_channels}→{module.out_channels}")
            if hasattr(module, "kernel_size"):
              if isinstance(module.kernel_size, tuple):
                kernel_str = f"k{module.kernel_size}"
              else:
                kernel_str = f"k{module.kernel_size}"
              layer_info_parts.append(kernel_str)
            if (
              hasattr(module, "stride")
              and module.stride != (1, 1)
              and module.stride != 1
            ):
              stride_str = (
                f"s{module.stride}"
                if isinstance(module.stride, int)
                else f"s{module.stride}"
              )
              layer_info_parts.append(stride_str)
          elif isinstance(module, nn.MultiheadAttention):
            layer_info_parts.append(f"heads={module.num_heads}, dim={module.embed_dim}")
          elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d, nn.LayerNorm)):
            if hasattr(module, "num_features"):
              layer_info_parts.append(f"features={module.num_features}")
            elif hasattr(module, "normalized_shape"):
              layer_info_parts.append(f"shape={module.normalized_shape}")

          layer_info = (
            f"{layer_type}({', '.join(layer_info_parts)})"
            if layer_info_parts
            else layer_type
          )
          param_str = f"{num_params:,}" if num_params > 0 else "-"
          print(f"  {name:<48} {layer_info:<43} {param_str:<15}")

    print("-" * 80)

  print("=" * 80 + "\n")


def compute_reconstruction_metrics(
  original: torch.Tensor, reconstructed: torch.Tensor, input_dim: int = 72
) -> dict[str, float]:
  """Compute detailed reconstruction metrics per feature type.

  Inputs are organized as:
  - joint_pos: 0-28 (29 dims)
  - joint_vel: 29-57 (29 dims)
  - anchor_pos: 58-60 (3 dims)
  - anchor_quat: 61-64 (4 dims)
  - object_pos: 65-67 (3 dims)
  - object_quat: 68-71 (4 dims)
  """
  mse = nn.MSELoss(reduction="none")
  mae = nn.L1Loss(reduction="none")

  # Compute per-feature-type losses
  errors = mse(original, reconstructed)  # (batch, horizon, input_dim)
  errors_mae = mae(original, reconstructed)

  # Overall metrics
  overall_mse = errors.mean().item()
  overall_mae = errors_mae.mean().item()

  # Per-feature-type metrics
  joint_pos_mse = errors[:, :, 0:29].mean().item()
  joint_pos_mae = errors_mae[:, :, 0:29].mean().item()

  joint_vel_mse = errors[:, :, 29:58].mean().item()
  joint_vel_mae = errors_mae[:, :, 29:58].mean().item()

  anchor_pos_mse = errors[:, :, 58:61].mean().item()
  anchor_pos_mae = errors_mae[:, :, 58:61].mean().item()

  anchor_quat_mse = errors[:, :, 61:65].mean().item()
  anchor_quat_mae = errors_mae[:, :, 61:65].mean().item()

  object_pos_mse = errors[:, :, 65:68].mean().item()
  object_pos_mae = errors_mae[:, :, 65:68].mean().item()

  object_quat_mse = errors[:, :, 68:72].mean().item()
  object_quat_mae = errors_mae[:, :, 68:72].mean().item()

  return {
    "reconstruction_mse": overall_mse,
    "reconstruction_mae": overall_mae,
    "joint_pos_mse": joint_pos_mse,
    "joint_pos_mae": joint_pos_mae,
    "joint_vel_mse": joint_vel_mse,
    "joint_vel_mae": joint_vel_mae,
    "anchor_pos_mse": anchor_pos_mse,
    "anchor_pos_mae": anchor_pos_mae,
    "anchor_quat_mse": anchor_quat_mse,
    "anchor_quat_mae": anchor_quat_mae,
    "object_pos_mse": object_pos_mse,
    "object_pos_mae": object_pos_mae,
    "object_quat_mse": object_quat_mse,
    "object_quat_mae": object_quat_mae,
  }


def train_epoch(
  model: TrajectoryAutoencoderBase,
  dataloader: DataLoader,
  optimizer: torch.optim.Optimizer,
  criterion: nn.Module,
  device: str,
  input_dim: int = 72,
  use_wandb: bool = False,
  epoch: int = 0,
  scheduler: torch.optim.lr_scheduler.LRScheduler | None = None,
  scheduler_name: str = "none",
  normalizer: TrajectoryNormalizer | None = None,
) -> tuple[float, dict[str, float]]:
  """Train for one epoch."""
  model.train()
  total_loss = 0.0
  all_metrics = {
    "reconstruction_mse": 0.0,
    "reconstruction_mae": 0.0,
    "joint_pos_mse": 0.0,
    "joint_pos_mae": 0.0,
    "joint_vel_mse": 0.0,
    "joint_vel_mae": 0.0,
    "anchor_pos_mse": 0.0,
    "anchor_pos_mae": 0.0,
    "anchor_quat_mse": 0.0,
    "anchor_quat_mae": 0.0,
    "object_pos_mse": 0.0,
    "object_pos_mae": 0.0,
    "object_quat_mse": 0.0,
    "object_quat_mae": 0.0,
  }
  num_batches = 0

  for batch_idx, batch in enumerate(dataloader):
    # Normalize batch once if normalizer is present (for loss computation)
    if hasattr(model, "normalizer") and model.normalizer is not None:
      batch_normalized = model.normalizer(batch)
    else:
      batch_normalized = batch

    # Forward pass - model normalizes internally, but we use pre-normalized batch for loss
    reconstructed_normalized, latent = model(batch)

    # Compute loss in normalized space
    loss = criterion(reconstructed_normalized, batch_normalized)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Update scheduler if per-batch scheduler (OneCycleLR or warmup_cosine)
    # Note: Step AFTER optimizer.step() so the scheduler updates for the next iteration
    if scheduler_name in ["onecycle", "warmup_cosine"] and scheduler is not None:
      scheduler.step()
      # Debug: Print LR for first few batches of first epoch
      if epoch == 0 and batch_idx < 3:
        current_lr = optimizer.param_groups[0]["lr"]
        print(
          f"  [Debug] Epoch {epoch + 1}, Batch {batch_idx + 1}: LR = {current_lr:.6e}, Scheduler step = {scheduler.last_epoch}"
        )

    # Accumulate metrics using normalized values
    batch_metrics = compute_reconstruction_metrics(
      batch_normalized, reconstructed_normalized, input_dim
    )
    total_loss += loss.item()
    for key in all_metrics:
      all_metrics[key] += batch_metrics[key]
    num_batches += 1

  # Average metrics
  avg_loss = total_loss / num_batches
  for key in all_metrics:
    all_metrics[key] /= num_batches

  return avg_loss, all_metrics


def validate(
  model: TrajectoryAutoencoderBase,
  dataloader: DataLoader,
  criterion: nn.Module,
  device: str,
  input_dim: int = 72,
  normalizer: TrajectoryNormalizer | None = None,
) -> tuple[float, dict[str, float], float]:
  """Validate the model.

  Returns:
      Tuple of (average_loss, metrics_dict, inference_time_seconds)
  """
  model.eval()
  total_loss = 0.0
  all_metrics = {
    "reconstruction_mse": 0.0,
    "reconstruction_mae": 0.0,
    "joint_pos_mse": 0.0,
    "joint_pos_mae": 0.0,
    "joint_vel_mse": 0.0,
    "joint_vel_mae": 0.0,
    "anchor_pos_mse": 0.0,
    "anchor_pos_mae": 0.0,
    "anchor_quat_mse": 0.0,
    "anchor_quat_mae": 0.0,
    "object_pos_mse": 0.0,
    "object_pos_mae": 0.0,
    "object_quat_mse": 0.0,
    "object_quat_mae": 0.0,
  }
  num_batches = 0

  # Measure inference time
  inference_start = time.time()
  if device == "cuda" and torch.cuda.is_available():
    torch.cuda.synchronize()

  with torch.no_grad():
    for batch in dataloader:
      # Normalize batch once if normalizer is present (for loss computation)
      if hasattr(model, "normalizer") and model.normalizer is not None:
        batch_normalized = model.normalizer(batch)
      else:
        batch_normalized = batch

      # Forward pass - model normalizes internally, but we use pre-normalized batch for loss
      reconstructed_normalized, latent = model(batch)

      # Compute loss in normalized space
      loss = criterion(reconstructed_normalized, batch_normalized)

      # Accumulate metrics using normalized values
      batch_metrics = compute_reconstruction_metrics(
        batch_normalized, reconstructed_normalized, input_dim
      )
      total_loss += loss.item()
      for key in all_metrics:
        all_metrics[key] += batch_metrics[key]
      num_batches += 1

  # Synchronize GPU operations before measuring end time
  if device == "cuda" and torch.cuda.is_available():
    torch.cuda.synchronize()
  inference_end = time.time()
  inference_time = inference_end - inference_start

  # Average metrics
  avg_loss = total_loss / num_batches
  for key in all_metrics:
    all_metrics[key] /= num_batches

  return avg_loss, all_metrics, inference_time


@hydra.main(
  version_base=None,
  config_path="../../../conf/trajectory_autoencoder",
  config_name="config",
)
def main(cfg: DictConfig) -> None:
  """Main training function with Hydra configuration."""
  # Convert config to dict for easier access (avoids struct mode issues)
  # Use resolve=True to ensure all values are resolved (including from base configs)
  cfg_dict = OmegaConf.to_container(cfg, resolve=True)
  assert isinstance(cfg_dict, dict)

  # Debug: Print config keys to verify base config is loaded
  if "motion_dir" not in cfg_dict and "wandb_registry" not in cfg_dict:
    print("WARNING: Neither motion_dir nor wandb_registry found in config!")
    print(f"Available keys: {list(cfg_dict.keys())[:20]}")
    print(f"Full config: {OmegaConf.to_yaml(cfg)}")

  # Helper function to get config values safely
  # Prefer dict access since it's already resolved and includes merged values
  def get_cfg(key: str, default=None):
    # First try dict access (most reliable for merged configs)
    if isinstance(cfg_dict, dict):
      keys = key.split(".")
      current = cfg_dict
      for k in keys:
        if isinstance(current, dict) and k in current:
          current = current[k]
        else:
          # Fallback to OmegaConf.select if not in dict
          value = OmegaConf.select(cfg, key, default=default)
          if value is None and default is not None:
            return default
          return value
      # Return None if value is None and no default provided
      if current is None and default is not None:
        return default
      return current
    # Fallback to OmegaConf.select
    return OmegaConf.select(cfg, key, default=default)

  # Set device if not specified or if CUDA not available
  device = get_cfg("device", "cuda")
  if device == "cuda" and not torch.cuda.is_available():
    print("CUDA not available, using CPU")
    device = "cpu"

  # Use Hydra's output directory (which is <output_dir>/<date>/<time>)
  # This ensures logs and model checkpoints are in the same directory
  hydra_cfg = HydraConfig.get()
  output_dir = Path(hydra_cfg.run.dir)
  print(f"Output directory: {output_dir}")

  # Get architecture from config name (e.g., "unet", "transformer", "tcn")
  architecture = hydra_cfg.job.config_name
  if architecture is None:
    raise ValueError("Config name is None. Please specify a valid config name.")
  print(f"Architecture: {architecture}")

  # Load robot and compute body indexes like in commands.py
  body_names = get_cfg("body_names", [])
  assert isinstance(body_names, list), (
    f"body_names must be a list, got {type(body_names)}"
  )
  assert all(isinstance(name, str) for name in body_names), (
    "All body_names must be strings"
  )
  robot_cfg = get_g1_robot_cfg()
  robot = Entity(robot_cfg)
  body_indexes = torch.tensor(
    robot.find_bodies(body_names, preserve_order=True)[0],
    dtype=torch.long,
    device=device,
  )
  # anchor_body_index should be the index within body_names (motion body names), not robot.body_names
  anchor_body_name = get_cfg("anchor_body_name", "torso_link")
  if anchor_body_name not in body_names:
    raise ValueError(
      f"anchor_body_name '{anchor_body_name}' must be in body_names {body_names}"
    )
  anchor_body_index = body_names.index(anchor_body_name)

  # Handle wandb download or local motion directory (simplified logic similar to commands.py)
  motion_dir = get_cfg("motion_dir", None)
  wandb_entity = get_cfg("wandb_entity", None)
  wandb_project = get_cfg("wandb_project", None)

  # Support legacy wandb_registry format for backward compatibility
  wandb_registry = get_cfg("wandb_registry", None)
  if wandb_registry is not None and isinstance(wandb_registry, dict):
    if wandb_entity is None:
      wandb_entity = wandb_registry.get("entity")
    if wandb_project is None:
      wandb_project = wandb_registry.get("project")

  # Download motions from wandb if wandb_entity and wandb_project are provided
  if wandb_entity is not None and wandb_project is not None:
    if not WANDB_AVAILABLE:
      raise ImportError(
        "wandb is required to download motions. Install with: pip install wandb"
      )
    # Type assertions for type checker
    assert isinstance(wandb_entity, str) and isinstance(wandb_project, str)
    print(f"[INFO] Downloading motions from wandb: {wandb_entity}/{wandb_project}")
    motion_dir = str(
      download_motions_from_wandb(
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        artifact_type="motions",
      )
    )
    print(f"[INFO] Using downloaded motions from: {motion_dir}")
  elif motion_dir is not None:
    # Use local motion directory
    print(f"Using local motion directory: {motion_dir}")
  else:
    raise ValueError(
      "Either 'motion_dir' must be specified, or 'wandb_entity' and "
      "'wandb_project' must be set in config"
    )

  # Create dataset (load directly on GPU)
  print(f"Loading dataset from {motion_dir} on {device}...")
  dataset = TrajectoryDataset(
    motion_dir=str(motion_dir),
    traj_name_patterns=get_cfg("traj_patterns", [".*"]),
    body_indexes=body_indexes,
    anchor_body_index=anchor_body_index,
    horizon=get_cfg("horizon", 32),
    device=device,
    stride=get_cfg("stride", 1),
  )
  print(f"Loaded {len(dataset)} trajectory windows")

  # Split into train/val
  val_split = get_cfg("val_split", 0.2)
  val_size = int(len(dataset) * val_split)
  train_size = len(dataset) - val_size
  train_dataset, val_dataset = torch.utils.data.random_split(
    dataset, [train_size, val_size]
  )

  # Setup normalization
  normalizer: TrajectoryNormalizer | None = None
  mean_min = mean_max = std_min = std_max = 0.0  # Initialize for wandb logging
  if get_cfg("normalize", True):
    print("Computing normalization statistics from training data...")
    normalizer = TrajectoryNormalizer().to(device)

    # Collect all training data to compute statistics
    max_samples_for_stats = min(10000, len(train_dataset))
    indices = torch.randperm(len(train_dataset))[:max_samples_for_stats]

    samples = []
    for idx in indices:
      sample = train_dataset[idx]
      samples.append(sample)

    all_data = torch.stack(samples)
    normalizer.fit(all_data)

    print(f"Normalization statistics computed from {len(samples)} samples")
    mean_tensor = getattr(normalizer, "mean", None)
    std_tensor = getattr(normalizer, "std", None)
    if mean_tensor is not None and std_tensor is not None:
      mean_min = mean_tensor.min().item()
      mean_max = mean_tensor.max().item()
      std_min = std_tensor.min().item()
      std_max = std_tensor.max().item()
      print(f"  Mean range: [{mean_min:.4f}, {mean_max:.4f}]")
      print(f"  Std range: [{std_min:.4f}, {std_max:.4f}]")

    # Save normalizer
    normalizer_file = get_cfg("normalizer_file")
    normalizer_path = (
      Path(normalizer_file) if normalizer_file else (output_dir / "normalizer.pt")
    )
    torch.save(normalizer.state_dict(), normalizer_path)
    print(f"Normalizer saved to {normalizer_path}")

  # Create data loaders
  batch_size = get_cfg("batch_size", 32)
  train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=0,
    pin_memory=False,
  )
  val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=0,
    pin_memory=False,
  )

  # Create model based on architecture type
  input_dim = dataset.get_input_dim()
  horizon = get_cfg("horizon", 32)
  latent_dim = get_cfg("latent_dim", 128)

  # Get model-specific parameters from config
  model_cfg = get_cfg("model", {})
  if not isinstance(model_cfg, dict):
    model_cfg = {}

  # Get the config section for this architecture (use architecture name directly as key)
  # Handle special case: 2dcnn architectures use _2dcnn prefix in config
  config_key = (
    architecture if not architecture.startswith("2dcnn") else f"_{architecture}"
  )
  arch_cfg = model_cfg.get(config_key, {}) if isinstance(model_cfg, dict) else {}

  # Helper function to get model parameter with default
  def get_model_param(key: str, default: any = None) -> any:
    return arch_cfg.get(key, default) if isinstance(arch_cfg, dict) else default

  if architecture == "unet":
    model: TrajectoryAutoencoderBase = TrajectoryAutoencoderUNet(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_channels=get_model_param("encoder_channels"),
      decoder_channels=get_model_param("decoder_channels"),
      use_batch_norm=get_model_param("use_batch_norm", True),
    )
  elif architecture == "unet_residual":
    model: TrajectoryAutoencoderBase = TrajectoryAutoencoderUNetResidual(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_channels=get_model_param("encoder_channels"),
      decoder_channels=get_model_param("decoder_channels"),
      use_batch_norm=get_model_param("use_batch_norm", True),
    )
  elif architecture == "unet_simple":
    model: TrajectoryAutoencoderBase = TrajectoryAutoencoderUNetSimple(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_hidden_dims=get_model_param("encoder_hidden_dims"),
      decoder_hidden_dims=get_model_param("decoder_hidden_dims"),
      use_batch_norm=get_model_param("use_batch_norm", True),
      dropout=get_model_param("dropout", 0.0),
    )
  elif architecture == "unet_simple_residual":
    model: TrajectoryAutoencoderBase = TrajectoryAutoencoderUNetSimpleResidual(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_hidden_dims=get_model_param("encoder_hidden_dims"),
      decoder_hidden_dims=get_model_param("decoder_hidden_dims"),
      use_batch_norm=get_model_param("use_batch_norm", True),
      dropout=get_model_param("dropout", 0.0),
      use_residual=get_model_param("use_residual", True),
    )
  elif architecture == "unet_simple_temporal":
    model: TrajectoryAutoencoderBase = TrajectoryAutoencoderUNetSimpleTemporal(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_hidden_dims=get_model_param("encoder_hidden_dims"),
      decoder_hidden_dims=get_model_param("decoder_hidden_dims"),
      use_batch_norm=get_model_param("use_batch_norm", True),
      dropout=get_model_param("dropout", 0.0),
      use_temporal_pos_encoding=get_model_param("use_temporal_pos_encoding", True),
      use_temporal_conv=get_model_param("use_temporal_conv", False),
      temporal_conv_dim=get_model_param("temporal_conv_dim"),
      temporal_conv_kernel_size=get_model_param("temporal_conv_kernel_size", 3),
    )
  elif architecture == "tcn":
    model = TrajectoryAutoencoderTCN(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      num_channels=get_model_param("num_channels"),
      kernel_size=get_model_param("kernel_size", 3),
      dropout=get_model_param("dropout", 0.2),
    )
  elif architecture == "2dcnn":
    model = TrajectoryAutoencoder2DCNN(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_channels=get_model_param("encoder_channels"),
      decoder_channels=get_model_param("decoder_channels"),
      kernel_size=get_model_param("kernel_size", 3),
      use_batch_norm=get_model_param("use_batch_norm", True),
    )
  elif architecture == "2dcnn_causal":
    model = TrajectoryAutoencoder2DCNNCausal(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_channels=get_model_param("encoder_channels"),
      decoder_channels=get_model_param("decoder_channels"),
      kernel_size=get_model_param("kernel_size", 3),
      use_batch_norm=get_model_param("use_batch_norm", True),
    )
  elif architecture == "transformer":
    model = TrajectoryAutoencoderTransformer(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      d_model=get_model_param("d_model", 256),
      nhead=get_model_param("nhead", 8),
      num_encoder_layers=get_model_param("num_encoder_layers", 4),
      num_decoder_layers=get_model_param("num_decoder_layers", 4),
      dim_feedforward=get_model_param("dim_feedforward", 1024),
      dropout=get_model_param("dropout", 0.1),
      activation=get_model_param("activation", "relu"),
    )
  elif architecture == "fsq":
    model = TrajectoryAutoencoderFSQ(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      hidden_dim=get_model_param("hidden_dim", 256),
      fsq_dim=get_model_param("fsq_dim", 8),
      fsq_levels=get_model_param("fsq_levels", None),
      num_temporal_layers=get_model_param("num_temporal_layers", 2),
      use_batch_norm=get_model_param("use_batch_norm", True),
      dropout=get_model_param("dropout", 0.0),
    )
  else:
    raise ValueError(f"Unknown architecture: {architecture}")

  model = model.to(device)

  # Attach normalizer to model if available (so it's part of checkpoint)
  if normalizer is not None:
    model.normalizer = normalizer.to(device)
    print("Normalizer attached to model (will be saved in checkpoint)")

  # Calculate trainable parameters
  trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
  total_params = sum(p.numel() for p in model.parameters())

  print("Model created:")
  print(f"  Architecture: {architecture}")
  print(f"  Horizon: {horizon}")
  print(f"  Input dim: {input_dim}")
  print(f"  Latent dim: {latent_dim}")
  print(f"  Total parameters: {total_params:,}")
  print(f"  Trainable parameters: {trainable_params:,}")

  # Print detailed architecture
  batch_size_val = get_cfg("batch_size", 32)
  if isinstance(batch_size_val, int):
    batch_size = batch_size_val
  else:
    batch_size = 32
  if isinstance(horizon, int) and isinstance(input_dim, int):
    print_model_architecture(model, (batch_size, horizon, input_dim))

  # Initialize wandb
  no_wandb = get_cfg("no_wandb", False)
  use_wandb = WANDB_AVAILABLE and not no_wandb
  if use_wandb:
    assert wandb is not None, "wandb not available"
    wandb_project = get_cfg("wandb_project", "trajectory_autoencoder")

    # Get Hydra config name for wandb run name (reuse hydra_cfg from above)
    config_name = hydra_cfg.job.config_name if hydra_cfg.job.config_name else "config"

    # Use config name from Hydra, or fallback to config value, or use config name
    wandb_run_name = get_cfg("wandb_run_name")
    if not wandb_run_name:
      # Use Hydra config name as the experiment name, with latent_dim appended
      wandb_run_name = f"{config_name}_l{latent_dim}"

    cfg_dict_for_wandb = OmegaConf.to_container(cfg, resolve=True)
    # Convert to proper dict[str, Any] type for wandb
    if isinstance(cfg_dict_for_wandb, dict):
      wandb_config: dict[str, Any] = {str(k): v for k, v in cfg_dict_for_wandb.items()}
    else:
      wandb_config = {}
    wandb.init(
      project=wandb_project,
      name=wandb_run_name,
      config=wandb_config,
    )

    # Log model parameters to wandb config
    wandb.config.update(
      {
        "model/trainable_parameters": trainable_params,
        "model/total_parameters": total_params,
      }
    )

    if get_cfg("normalize", True):
      wandb.config.update(
        {
          "normalizer_mean_min": mean_min,
          "normalizer_mean_max": mean_max,
          "normalizer_std_min": std_min,
          "normalizer_std_max": std_max,
        }
      )

  # Loss and optimizer
  lr = get_cfg("lr", 0.001)
  criterion = nn.MSELoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=lr)

  # Setup learning rate scheduler
  steps_per_epoch = len(train_loader)
  epochs = get_cfg("epochs", 1000)
  total_steps = epochs * steps_per_epoch
  scheduler = None
  scheduler_type = "loss"  # Default for ReduceLROnPlateau
  scheduler_name = get_cfg("scheduler", "reduce_on_plateau")

  if scheduler_name == "onecycle":
    max_lr = get_cfg("max_lr")
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
      optimizer,
      max_lr=max_lr if max_lr else lr * 10,
      total_steps=total_steps,
      pct_start=0.3,
    )
    scheduler_type = "batch"
  elif scheduler_name == "cosine_annealing":
    min_lr = get_cfg("min_lr")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
      optimizer, T_max=epochs, eta_min=min_lr if min_lr else lr * 0.01
    )
    scheduler_type = "epoch"
  elif scheduler_name == "cosine_restarts":
    t_0 = get_cfg("t_0", 10)
    t_mult = get_cfg("t_mult", 2)
    min_lr = get_cfg("min_lr")
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
      optimizer,
      T_0=t_0,
      T_mult=t_mult,
      eta_min=min_lr if min_lr else lr * 0.01,
    )
    scheduler_type = "epoch"
  elif scheduler_name == "reduce_on_plateau":
    min_lr_val = get_cfg("min_lr")
    min_lr_for_scheduler = (
      min_lr_val if min_lr_val else lr * 0.0001
    )  # Default to 0.01% of base LR
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
      optimizer, mode="min", factor=0.5, patience=5, min_lr=min_lr_for_scheduler
    )
    scheduler_type = "loss"
    print("ReduceLROnPlateau scheduler initialized:")
    print(f"  Base LR: {lr}")
    print(f"  Min LR: {min_lr_for_scheduler}")
    print("  Patience: 5 epochs")
    print("  Factor: 0.5 (halves LR when triggered)")
  elif scheduler_name == "warmup_cosine":
    # Custom warmup + cosine annealing
    warmup_epochs = get_cfg("warmup_epochs", 10)
    min_lr_val = get_cfg("min_lr")
    min_lr_ratio = (min_lr_val / lr) if min_lr_val else 0.01  # Default to 1% of base LR

    warmup_steps = int(warmup_epochs * steps_per_epoch)
    cosine_steps = total_steps - warmup_steps

    def lr_lambda(step: int) -> float:
      if step < warmup_steps:
        # Warmup: linearly increase from 0 to 1
        return step / max(warmup_steps, 1)
      else:
        # Cosine annealing: decrease from 1 to min_lr_ratio
        progress = (step - warmup_steps) / max(cosine_steps, 1)
        # Cosine goes from 1 to -1, we want to map it to [min_lr_ratio, 1]
        # Formula: min_lr_ratio + (1 - min_lr_ratio) * 0.5 * (1 + cos(π * progress))
        cosine_factor = 0.5 * (1 + math.cos(math.pi * progress))
        return min_lr_ratio + (1 - min_lr_ratio) * cosine_factor

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scheduler_type = "batch"

    # Debug: Print scheduler info
    print("Warmup cosine scheduler initialized:")
    print(f"  Base LR: {lr}")
    print(f"  Min LR: {min_lr_val if min_lr_val else lr * 0.01}")
    print(f"  Min LR ratio: {min_lr_ratio:.6f}")
    print(f"  Warmup epochs: {warmup_epochs}")
    print(f"  Warmup steps: {warmup_steps}")
    print(f"  Total steps: {total_steps}")
    print(f"  Cosine steps: {cosine_steps}")
    # Check initial LR after scheduler creation
    initial_lr = optimizer.param_groups[0]["lr"]
    print(f"  Initial LR after scheduler init: {initial_lr:.6e}")

  # Early stopping configuration
  early_stopping_cfg = get_cfg("early_stopping", {})
  early_stopping_enabled = (
    early_stopping_cfg.get("enabled", False)
    if isinstance(early_stopping_cfg, dict)
    else False
  )
  early_stopping_patience = (
    early_stopping_cfg.get("patience", 20)
    if isinstance(early_stopping_cfg, dict)
    else 20
  )
  early_stopping_min_delta = (
    early_stopping_cfg.get("min_delta", 0.0)
    if isinstance(early_stopping_cfg, dict)
    else 0.0
  )
  restore_best_weights = (
    early_stopping_cfg.get("restore_best_weights", True)
    if isinstance(early_stopping_cfg, dict)
    else True
  )

  # Early stopping state
  best_val_loss = float("inf")
  epochs_without_improvement = 0
  best_model_state = None
  best_optimizer_state = None
  best_epoch = 0

  if early_stopping_enabled:
    print("\nEarly stopping enabled:")
    print(f"  Patience: {early_stopping_patience} epochs")
    print(f"  Min delta: {early_stopping_min_delta}")
    print(f"  Restore best weights: {restore_best_weights}")

  print("\nStarting training...")
  for epoch in range(epochs):
    train_loss, train_metrics = train_epoch(
      model,
      train_loader,
      optimizer,
      criterion,
      device,
      input_dim,
      use_wandb,
      epoch,
      scheduler
      if scheduler_name == "onecycle" or scheduler_name == "warmup_cosine"
      else None,
      scheduler_name,
      normalizer,
    )
    val_loss, val_metrics, val_inference_time = validate(
      model, val_loader, criterion, device, input_dim, normalizer
    )

    current_lr = optimizer.param_groups[0]["lr"]

    # Update scheduler based on type
    if scheduler_type == "loss":
      assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
      scheduler.step(val_loss)

    elif scheduler_name not in ["onecycle", "warmup_cosine"]:
      if scheduler is not None and not isinstance(
        scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
      ):
        scheduler.step()

    # Log to wandb
    if use_wandb and wandb is not None:
      log_dict = {
        "epoch": epoch + 1,
        "train/loss": train_loss,
        "val/loss": val_loss,
        "learning_rate": current_lr,
        "val/inference_time_seconds": val_inference_time,
      }
      if early_stopping_enabled:
        log_dict["early_stopping/epochs_without_improvement"] = (
          epochs_without_improvement
        )
        log_dict["early_stopping/best_val_loss"] = best_val_loss
      for key, value in train_metrics.items():
        if key != "loss":
          log_dict[f"train/{key}"] = value
      for key, value in val_metrics.items():
        if key != "loss":
          log_dict[f"val/{key}"] = value
      wandb.log(log_dict)

    print(
      f"Epoch {epoch + 1}/{epochs}: "
      f"train_loss={train_loss:.6f}, val_loss={val_loss:.6f}, lr={current_lr:.2e}"
    )
    if epoch == 0 or (epoch + 1) % 10 == 0:
      print(
        f"  Train MSE: {train_metrics['reconstruction_mse']:.6f}, "
        f"MAE: {train_metrics['reconstruction_mae']:.6f}"
      )
      print(
        f"  Val MSE: {val_metrics['reconstruction_mse']:.6f}, "
        f"MAE: {val_metrics['reconstruction_mae']:.6f}"
      )

    # Check for improvement and early stopping
    is_best = False

    if val_loss < best_val_loss - early_stopping_min_delta:
      # Significant improvement
      best_val_loss = val_loss
      best_epoch = epoch
      epochs_without_improvement = 0
      is_best = True

      # Save best model state for potential restoration
      if restore_best_weights:
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_optimizer_state = {
          k: v.cpu().clone() if isinstance(v, torch.Tensor) else v
          for k, v in optimizer.state_dict().items()
        }
    else:
      # No improvement
      epochs_without_improvement += 1

    # Save checkpoint
    if is_best:
      checkpoint_path = output_dir / "best_model.pt"
      # Normalizer is now part of model.state_dict(), but keep separate entry for backward compatibility
      checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),  # Includes normalizer if attached
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "train_loss": train_loss,
        "horizon": horizon,
        "latent_dim": latent_dim,
        "input_dim": input_dim,
        "architecture": architecture,
      }
      # Keep normalizer_state_dict for backward compatibility with old checkpoints
      if normalizer is not None:
        checkpoint_dict["normalizer_state_dict"] = normalizer.state_dict()
      else:
        checkpoint_dict["normalizer_state_dict"] = None
      #   torch.save(checkpoint_dict, checkpoint_path)
      #   print(
      #     f"  ✓ New best model! (improvement: {improvement:.6f}) Saved to {checkpoint_path}"
      #   )

      # Optionally save JIT-compiled encoder
      save_jit = get_cfg("save_jit", False)
      if save_jit:
        jit_path = output_dir / "best_model.jit"
        save_jitted_encoder(model, jit_path, device)

    # Early stopping check
    if early_stopping_enabled and epochs_without_improvement >= early_stopping_patience:
      print("\nEarly stopping triggered!")
      print(f"  No improvement for {epochs_without_improvement} epochs")
      print(f"  Best validation loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")
      print(f"  Current validation loss: {val_loss:.6f}")

      # Restore best weights if requested
      if restore_best_weights and best_model_state is not None:
        print(f"  Restoring best model weights from epoch {best_epoch + 1}...")
        model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
        optimizer.load_state_dict(best_optimizer_state)
        # Move optimizer states back to device
        for state in optimizer.state.values():
          for k, v in state.items():
            if isinstance(v, torch.Tensor):
              state[k] = v.to(device)
        print("  Best model weights restored.")

      # Log early stopping to wandb
      if use_wandb and wandb is not None:
        wandb.log(
          {
            "early_stopping/triggered": True,
            "early_stopping/best_epoch": best_epoch + 1,
            "early_stopping/best_val_loss": best_val_loss,
            "early_stopping/epochs_without_improvement": epochs_without_improvement,
          }
        )

      break

    # Save periodic checkpoint
    if (epoch + 1) % 50 == 0:
      checkpoint_path = output_dir / f"checkpoint_epoch_{epoch + 1}.pt"
      # Normalizer is now part of model.state_dict(), but keep separate entry for backward compatibility
      checkpoint_dict = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),  # Includes normalizer if attached
        "optimizer_state_dict": optimizer.state_dict(),
        "val_loss": val_loss,
        "train_loss": train_loss,
      }
      # Keep normalizer_state_dict for backward compatibility with old checkpoints
      if normalizer is not None:
        checkpoint_dict["normalizer_state_dict"] = normalizer.state_dict()
      else:
        checkpoint_dict["normalizer_state_dict"] = None
      torch.save(checkpoint_dict, checkpoint_path)

  print("\nTraining complete!")
  print(f"Best validation loss: {best_val_loss:.6f} (epoch {best_epoch + 1})")

  # Restore best weights if training completed normally (not early stopped) and restore is enabled
  # Note: If early stopping was triggered, weights were already restored in the loop
  if restore_best_weights and best_model_state is not None:
    # Check if we completed all epochs (didn't early stop)
    if epoch == epochs - 1:  # Completed all epochs
      print(f"Restoring best model weights from epoch {best_epoch + 1}...")
      model.load_state_dict({k: v.to(device) for k, v in best_model_state.items()})
      if best_optimizer_state is not None:
        optimizer.load_state_dict(best_optimizer_state)
        # Move optimizer states back to device
        for state in optimizer.state.values():
          for k, v in state.items():
            if isinstance(v, torch.Tensor):
              state[k] = v.to(device)
      print("Best model weights restored.")

  if use_wandb and wandb is not None:
    wandb.finish()
    print("Wandb run finished.")


if __name__ == "__main__":
  main()
