"""Visualize reconstructed trajectories from trained trajectory autoencoder.

This script loads a trained model checkpoint and visualizes original vs reconstructed
trajectories, showing how well the model reconstructs different feature types.

Example usage:
    # Minimal usage (loads all parameters from training config):
    python scripts/visualize_reconstruction.py \\
        --checkpoint logs/trajectory_autoencoder/unet_simple_temporal/best_model.pt \\
        --num-samples 5
    
    # With custom features:
    python scripts/visualize_reconstruction.py \\
        --checkpoint logs/trajectory_autoencoder/unet_simple_temporal/best_model.pt \\
        --num-samples 5 \\
        --features joint_pos joint_vel object_pos

The script will:
1. Load the model checkpoint and normalizer
2. Load trajectory data from the specified directory
3. Reconstruct trajectories using the model
4. Generate plots comparing original vs reconstructed for each feature group
5. Save plots to checkpoint_dir/visualizations/ (or --output-dir)
6. Print and save summary metrics
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from mjlab.asset_zoo.robots import get_g1_robot_cfg
from mjlab.entity import Entity
from mjlab.models import (
  TrajectoryAutoencoder2DCNN,
  TrajectoryAutoencoder2DCNNCausal,
  TrajectoryAutoencoderBase,
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

# Feature ranges for visualization
FEATURE_RANGES = {
  "joint_pos": (0, 29),
  "joint_vel": (29, 58),
  "anchor_pos": (58, 61),
  "anchor_quat": (61, 65),
  "object_pos": (65, 68),
  "object_quat": (68, 72),
}


def load_training_config(checkpoint_path: Path) -> dict:
  """Load training configuration from Hydra config file.

  Args:
      checkpoint_path: Path to checkpoint file

  Returns:
      Dictionary with training config values
  """
  checkpoint_dir = checkpoint_path.parent
  config_yaml = checkpoint_dir / ".hydra" / "config.yaml"

  if not config_yaml.exists():
    return {}

  try:
    import yaml

    with open(config_yaml, "r") as f:
      hydra_cfg = yaml.safe_load(f)
    return hydra_cfg
  except Exception as e:
    print(f"  Warning: Could not load config from {config_yaml}: {e}")
    return {}


def load_model_from_checkpoint(
  checkpoint_path: Path,
  device: str = "cpu",
) -> tuple[TrajectoryAutoencoderBase, TrajectoryNormalizer | None]:
  """Load model and normalizer from checkpoint.

  Args:
      checkpoint_path: Path to model checkpoint (.pt file)
      device: Device to load model on

  Returns:
      Tuple of (model, normalizer)
  """
  print(f"Loading checkpoint from: {checkpoint_path}")
  checkpoint = torch.load(checkpoint_path, map_location=device)

  # Extract model parameters
  architecture = checkpoint.get("architecture", "unet_simple")
  horizon = checkpoint["horizon"]
  input_dim = checkpoint["input_dim"]
  latent_dim = checkpoint["latent_dim"]

  print(f"  Architecture: {architecture}")
  print(f"  Horizon: {horizon}")
  print(f"  Input dim: {input_dim}")
  print(f"  Latent dim: {latent_dim}")

  # Create model based on architecture
  # Try to load model config from checkpoint directory (Hydra config)
  model_cfg = {}
  checkpoint_dir = checkpoint_path.parent

  # Try Hydra config first
  config_yaml = checkpoint_dir / ".hydra" / "config.yaml"
  if config_yaml.exists():
    try:
      import yaml

      with open(config_yaml, "r") as f:
        hydra_cfg = yaml.safe_load(f)
        model_cfg = hydra_cfg.get("model", {})
        print(f"  Loaded model config from: {config_yaml}")
    except Exception as e:
      print(f"  Warning: Could not load config from {config_yaml}: {e}")

  # Fallback to checkpoint model_config if available
  if not model_cfg:
    model_cfg = checkpoint.get("model_config", {})

  # If still no config, use defaults (will be None and model will use its defaults)
  if not model_cfg:
    print("  Using default model parameters (no config found)")

  if architecture == "unet":
    model = TrajectoryAutoencoderUNet(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_channels=model_cfg.get("encoder_channels"),
      decoder_channels=model_cfg.get("decoder_channels"),
      use_batch_norm=model_cfg.get("use_batch_norm", True),
    )
  elif architecture == "unet_residual":
    model = TrajectoryAutoencoderUNetResidual(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_channels=model_cfg.get("encoder_channels"),
      decoder_channels=model_cfg.get("decoder_channels"),
      use_batch_norm=model_cfg.get("use_batch_norm", True),
    )
  elif architecture == "unet_simple":
    model = TrajectoryAutoencoderUNetSimple(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_hidden_dims=model_cfg.get("encoder_hidden_dims"),
      decoder_hidden_dims=model_cfg.get("decoder_hidden_dims"),
      use_batch_norm=model_cfg.get("use_batch_norm", True),
      dropout=model_cfg.get("dropout", 0.0),
    )
  elif architecture == "unet_simple_residual":
    model = TrajectoryAutoencoderUNetSimpleResidual(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_hidden_dims=model_cfg.get("encoder_hidden_dims"),
      decoder_hidden_dims=model_cfg.get("decoder_hidden_dims"),
      use_batch_norm=model_cfg.get("use_batch_norm", True),
      dropout=model_cfg.get("dropout", 0.0),
      use_residual=model_cfg.get("use_residual", True),
    )
  elif architecture == "unet_simple_temporal":
    model = TrajectoryAutoencoderUNetSimpleTemporal(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_hidden_dims=model_cfg.get("encoder_hidden_dims"),
      decoder_hidden_dims=model_cfg.get("decoder_hidden_dims"),
      use_batch_norm=model_cfg.get("use_batch_norm", True),
      dropout=model_cfg.get("dropout", 0.0),
      use_temporal_pos_encoding=model_cfg.get("use_temporal_pos_encoding", True),
      use_temporal_conv=model_cfg.get("use_temporal_conv", False),
      temporal_conv_dim=model_cfg.get("temporal_conv_dim"),
      temporal_conv_kernel_size=model_cfg.get("temporal_conv_kernel_size", 3),
    )
  elif architecture == "tcn":
    model = TrajectoryAutoencoderTCN(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      num_channels=model_cfg.get("num_channels"),
      kernel_size=model_cfg.get("kernel_size", 3),
      dropout=model_cfg.get("dropout", 0.2),
    )
  elif architecture == "2dcnn":
    model = TrajectoryAutoencoder2DCNN(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_channels=model_cfg.get("encoder_channels"),
      decoder_channels=model_cfg.get("decoder_channels"),
      kernel_size=model_cfg.get("kernel_size", 3),
      use_batch_norm=model_cfg.get("use_batch_norm", True),
    )
  elif architecture == "2dcnn_causal":
    model = TrajectoryAutoencoder2DCNNCausal(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_channels=model_cfg.get("encoder_channels"),
      decoder_channels=model_cfg.get("decoder_channels"),
      kernel_size=model_cfg.get("kernel_size", 3),
      use_batch_norm=model_cfg.get("use_batch_norm", True),
    )
  elif architecture == "transformer":
    model = TrajectoryAutoencoderTransformer(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      d_model=model_cfg.get("d_model", 256),
      nhead=model_cfg.get("nhead", 8),
      num_encoder_layers=model_cfg.get("num_encoder_layers", 4),
      num_decoder_layers=model_cfg.get("num_decoder_layers", 4),
      dim_feedforward=model_cfg.get("dim_feedforward", 1024),
      dropout=model_cfg.get("dropout", 0.1),
    )
  else:
    raise ValueError(f"Unknown architecture: {architecture}")

  # Load model weights
  model.load_state_dict(checkpoint["model_state_dict"])
  model = model.to(device)
  model.eval()

  # Load normalizer if available
  normalizer = None
  if checkpoint.get("normalizer_state_dict") is not None:
    normalizer = TrajectoryNormalizer().to(device)
    normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
    print("  Normalizer loaded from checkpoint")
  else:
    # Try to load from separate normalizer file
    normalizer_path = checkpoint_path.parent / "normalizer.pt"
    if normalizer_path.exists():
      normalizer = TrajectoryNormalizer().to(device)
      normalizer.load_state_dict(torch.load(normalizer_path, map_location=device))
      print(f"  Normalizer loaded from: {normalizer_path}")

  return model, normalizer


def plot_feature_comparison(
  original: np.ndarray,
  reconstructed: np.ndarray,
  feature_name: str,
  feature_range: tuple[int, int],
  save_path: Path | None = None,
  show_plot: bool = True,
) -> None:
  """Plot original vs reconstructed trajectories for a feature group.

  Args:
      original: Original trajectory of shape (horizon, input_dim)
      reconstructed: Reconstructed trajectory of shape (horizon, input_dim)
      feature_name: Name of the feature group
      feature_range: (start_idx, end_idx) for this feature group
      save_path: Optional path to save the plot
      show_plot: Whether to display the plot
  """
  start_idx, end_idx = feature_range
  orig_features = original[:, start_idx:end_idx]  # (horizon, feature_dim)
  recon_features = reconstructed[:, start_idx:end_idx]  # (horizon, feature_dim)
  horizon = orig_features.shape[0]
  feature_dim = orig_features.shape[1]

  # Create subplots: one per feature dimension
  num_cols = min(4, feature_dim)
  num_rows = (feature_dim + num_cols - 1) // num_cols

  fig, axes = plt.subplots(num_rows, num_cols, figsize=(4 * num_cols, 3 * num_rows))
  if feature_dim == 1:
    axes = [axes]
  else:
    axes = axes.flatten()

  time_steps = np.arange(horizon)

  for i in range(feature_dim):
    ax = axes[i]
    ax.plot(time_steps, orig_features[:, i], label="Original", linewidth=2, alpha=0.7)
    ax.plot(
      time_steps,
      recon_features[:, i],
      label="Reconstructed",
      linewidth=2,
      alpha=0.7,
      linestyle="--",
    )
    ax.set_title(f"{feature_name} [{i}]")
    ax.set_xlabel("Time step")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True, alpha=0.3)

  # Hide unused subplots
  for i in range(feature_dim, len(axes)):
    axes[i].axis("off")

  plt.suptitle(
    f"Original vs Reconstructed: {feature_name}", fontsize=14, fontweight="bold"
  )
  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved plot to: {save_path}")

  if show_plot:
    plt.show()
  else:
    plt.close()


def compute_reconstruction_metrics(
  original: torch.Tensor,
  reconstructed: torch.Tensor,
) -> dict[str, float]:
  """Compute reconstruction metrics per feature type.

  Args:
      original: Original trajectory of shape (batch_size, horizon, input_dim)
      reconstructed: Reconstructed trajectory of shape (batch_size, horizon, input_dim)

  Returns:
      Dictionary of metrics
  """
  mse = nn.MSELoss(reduction="none")
  mae = nn.L1Loss(reduction="none")

  errors_mse = mse(original, reconstructed)  # (batch, horizon, input_dim)
  errors_mae = mae(original, reconstructed)

  metrics = {
    "overall_mse": errors_mse.mean().item(),
    "overall_mae": errors_mae.mean().item(),
  }

  # Per-feature-type metrics
  for name, (start, end) in FEATURE_RANGES.items():
    metrics[f"{name}_mse"] = errors_mse[:, :, start:end].mean().item()
    metrics[f"{name}_mae"] = errors_mae[:, :, start:end].mean().item()

  return metrics


def visualize_reconstructions(
  checkpoint_path: str | Path,
  num_samples: int = 5,
  output_dir: str | Path | None = None,
  device: str = "cpu",
  show_plots: bool = True,
  feature_groups: list[str] | None = None,
) -> None:
  """Visualize reconstructed trajectories.

  Args:
      checkpoint_path: Path to model checkpoint
      num_samples: Number of samples to visualize
      output_dir: Directory to save plots (if None, saves next to checkpoint)
      device: Device to use
      show_plots: Whether to display plots
      feature_groups: List of feature groups to plot (if None, plots all)
  """
  checkpoint_path_obj = Path(checkpoint_path)
  if not checkpoint_path_obj.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path_obj}")

  # Load training config from Hydra config file
  training_cfg = load_training_config(checkpoint_path_obj)

  # Load all required parameters from config
  motion_dir = training_cfg.get("motion_dir")
  if motion_dir is None:
    raise ValueError("motion_dir not found in training config")
  print(f"  Loaded motion_dir from config: {motion_dir}")

  traj_name_patterns = training_cfg.get("traj_patterns", [".*"])
  print(f"  Loaded traj_patterns from config: {traj_name_patterns}")

  anchor_body_name = training_cfg.get("anchor_body_name", "torso_link")
  print(f"  Loaded anchor_body_name from config: {anchor_body_name}")

  body_names_list = training_cfg.get("body_names", [])
  if not body_names_list:
    raise ValueError("body_names not found in training config")
  body_names = tuple(body_names_list)
  print(f"  Loaded body_names from config: {len(body_names)} bodies")

  # Load model
  model, normalizer = load_model_from_checkpoint(checkpoint_path_obj, device=device)

  # Load robot and compute body indexes like in train_trajectory_autoencoder_hydra.py
  assert body_names is not None, "body_names must be provided or loaded from config"
  assert anchor_body_name is not None, (
    "anchor_body_name must be provided or loaded from config"
  )
  assert traj_name_patterns is not None, (
    "traj_name_patterns must be provided or loaded from config"
  )

  body_names_list = list(body_names)
  robot_cfg = get_g1_robot_cfg()
  robot = Entity(robot_cfg)
  body_indexes = torch.tensor(
    robot.find_bodies(body_names_list, preserve_order=True)[0],
    dtype=torch.long,
    device=device,
  )
  # anchor_body_index should be the index within body_names (motion body names), not robot.body_names
  anchor_body_index = body_names_list.index(anchor_body_name)

  # Create dataset
  horizon = model.horizon
  dataset = TrajectoryDataset(
    motion_dir=motion_dir,
    traj_name_patterns=traj_name_patterns,
    body_indexes=body_indexes,
    anchor_body_index=anchor_body_index,
    horizon=horizon,
    device=device,
    stride=1,
  )

  # Debug: Print dataset info
  print("\nDataset info:")
  print(f"  Number of windows: {len(dataset)}")
  print(f"  Horizon: {horizon}")
  print(f"  Motion loader has {dataset.motion_loader.num_motions} motions")

  # Check raw motion data
  if len(dataset) > 0:
    motion_loader = dataset.motion_loader
    if motion_loader.num_motions > 0:
      motion0 = motion_loader.motions[0]
      print("\n  Motion 0 info:")
      print(f"    time_step_total: {motion0.time_step_total}")
      print(f"    joint_pos shape: {motion0.joint_pos.shape}")
      print(
        f"    joint_pos range: [{motion0.joint_pos.min().item():.4f}, {motion0.joint_pos.max().item():.4f}]"
      )
      print(f"    joint_pos std: {motion0.joint_pos.std().item():.6f}")
      print(
        f"    body_pos_w shape: {motion0.body_pos_w.shape if hasattr(motion0, 'body_pos_w') else 'N/A'}"
      )
      if hasattr(motion0, "body_pos_w") and motion0.body_pos_w is not None:
        print(
          f"    body_pos_w range: [{motion0.body_pos_w.min().item():.4f}, {motion0.body_pos_w.max().item():.4f}]"
        )
        print(f"    body_pos_w std: {motion0.body_pos_w.std().item():.6f}")
        # Check if all timesteps are the same
        if motion0.body_pos_w.shape[0] > 1:
          first_t = motion0.body_pos_w[0]
          all_same = (motion0.body_pos_w == first_t.unsqueeze(0)).all().item()
          print(f"    All timesteps identical in body_pos_w: {all_same}")

    # Check first few windows
    print(f"\n  First 5 windows: {dataset.windows[: min(5, len(dataset))]}")

    # Debug: Check raw data slicing for first two windows
    motion0 = motion_loader.motions[0]
    window0 = dataset.windows[0]  # (0, 0)
    window1 = dataset.windows[1]  # (0, 1)
    print("\n  Debugging window extraction:")
    print(f"    Window 0: motion_idx={window0[0]}, start_t={window0[1]}")
    print(f"    Window 1: motion_idx={window1[0]}, start_t={window1[1]}")

    # Check raw joint_pos slices
    joint_slice0 = motion0.joint_pos[window0[1] : window0[1] + horizon]
    joint_slice1 = motion0.joint_pos[window1[1] : window1[1] + horizon]
    print(
      f"    joint_pos slice 0: std={joint_slice0.std().item():.6f}, first 3 values: {joint_slice0[0, :3].tolist()}"
    )
    print(
      f"    joint_pos slice 1: std={joint_slice1.std().item():.6f}, first 3 values: {joint_slice1[0, :3].tolist()}"
    )
    print(
      f"    joint_pos slices are different: {(joint_slice0 != joint_slice1).any().item()}"
    )
    print(
      f"    joint_pos slice 0 timesteps vary: {(joint_slice0[0] != joint_slice0[1]).any().item()}"
    )
    print(
      f"    joint_pos slice 1 timesteps vary: {(joint_slice1[0] != joint_slice1[1]).any().item()}"
    )

    # Check if consecutive timesteps in raw motion are different
    print("\n    Checking raw motion timesteps:")
    print(f"      motion0.joint_pos[0, :3]: {motion0.joint_pos[0, :3].tolist()}")
    print(f"      motion0.joint_pos[1, :3]: {motion0.joint_pos[1, :3].tolist()}")
    print(f"      motion0.joint_pos[2, :3]: {motion0.joint_pos[2, :3].tolist()}")
    print(
      f"      Timestep 0 != Timestep 1: {(motion0.joint_pos[0] != motion0.joint_pos[1]).any().item()}"
    )
    print(
      f"      Timestep 1 != Timestep 2: {(motion0.joint_pos[1] != motion0.joint_pos[2]).any().item()}"
    )

    # Check body_pos_w too
    if hasattr(motion0, "body_pos_w") and motion0.body_pos_w is not None:
      print(
        f"      motion0.body_pos_w[0, 0, :]: {motion0.body_pos_w[0, 0, :].tolist()}"
      )
      print(
        f"      motion0.body_pos_w[1, 0, :]: {motion0.body_pos_w[1, 0, :].tolist()}"
      )
      print(
        f"      body_pos_w Timestep 0 != Timestep 1: {(motion0.body_pos_w[0] != motion0.body_pos_w[1]).any().item()}"
      )

    # Check raw body_pos_w slices
    if hasattr(motion0, "body_pos_w") and motion0.body_pos_w is not None:
      body_slice0 = motion0.body_pos_w[window0[1] : window0[1] + horizon]
      body_slice1 = motion0.body_pos_w[window1[1] : window1[1] + horizon]
      print(f"    body_pos_w slice 0: std={body_slice0.std().item():.6f}")
      print(f"    body_pos_w slice 1: std={body_slice1.std().item():.6f}")
      print(
        f"    body_pos_w slices are different: {(body_slice0 != body_slice1).any().item()}"
      )
      print(
        f"    body_pos_w slice 0 timesteps vary: {(body_slice0[0] != body_slice0[1]).any().item()}"
      )

    # Check if windows are different
    sample0 = dataset[0]
    print(f"\n  Sample 0 shape: {sample0.shape}")
    print(f"  Sample 0 range: [{sample0.min().item():.4f}, {sample0.max().item():.4f}]")
    print(f"  Sample 0 std: {sample0.std().item():.6f}")

    if len(dataset) > 1:
      sample1 = dataset[1]
      diff = (sample0 != sample1).any().item()
      print(f"  First two samples are different: {diff}")
      if not diff:
        print("  WARNING: First two samples are identical!")
    # Check if first sample has variation across timesteps
    first_timestep = sample0[0]
    all_timesteps_same = (
      (sample0 == first_timestep.unsqueeze(0)).all(dim=1).all().item()
    )
    if all_timesteps_same:
      print("  ERROR: All timesteps in first sample are identical!")
      print("  This indicates the motion data is static (all timesteps are the same).")
      print(f"  First timestep values (first 10): {sample0[0, :10].tolist()}")
      # Check individual components
      print("  Checking components in sample0:")
      print(f"    joint_pos[0:5, 0] in sample0: {sample0[0:5, 0].tolist()}")
      print(f"    joint_vel[0:5, 0] in sample0: {sample0[0:5, 29].tolist()}")
      print("\n  This is a DATA QUALITY ISSUE, not a code bug.")
      print(f"  The motion files in {motion_dir} contain static data.")
      print("  Please regenerate your motion files with temporal variation.")
      raise ValueError(
        "Motion data is static (all timesteps identical). "
        "Cannot visualize reconstructions with static data. "
        "Please check your motion files and regenerate them with temporal variation."
      )
    else:
      timestep_variation = sample0.std(dim=0).mean().item()
      print(
        f"  Timestep variation in first sample (std mean): {timestep_variation:.6f}"
      )

  # Create data loader
  dataloader = DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)

  # Setup output directory
  if output_dir is None:
    output_dir_path = checkpoint_path_obj.parent / "visualizations"
  else:
    output_dir_path = Path(output_dir)
  output_dir_path.mkdir(parents=True, exist_ok=True)

  # Determine which features to plot
  if feature_groups is None:
    feature_groups = list(FEATURE_RANGES.keys())
  else:
    # Validate feature groups
    for fg in feature_groups:
      if fg not in FEATURE_RANGES:
        raise ValueError(
          f"Unknown feature group: {fg}. Available: {list(FEATURE_RANGES.keys())}"
        )

  print(f"\nVisualizing {num_samples} samples...")
  print(f"Output directory: {output_dir_path}")

  all_metrics = []

  with torch.no_grad():
    for sample_idx, batch in enumerate(dataloader):
      if sample_idx >= num_samples:
        break

      # Debug: Print which window we're using
      window_idx = sample_idx
      if window_idx < len(dataset):
        motion_idx, start_t = dataset.windows[window_idx]
        print(f"\nSample {sample_idx + 1}: motion_idx={motion_idx}, start_t={start_t}")

      # Check batch shape and content
      print(f"  Batch shape: {batch.shape}")
      if batch.shape[0] == 1:
        batch_data = batch[0]  # Remove batch dimension: (horizon, 72)
        # Check if all timesteps are the same
        first_timestep = batch_data[0]
        all_same = (batch_data == first_timestep.unsqueeze(0)).all(dim=1).all().item()
        if all_same:
          print("  WARNING: All timesteps in this sample are identical!")
        else:
          # Check variation across timesteps
          timestep_variation = batch_data.std(dim=0).mean().item()
          print(f"  Timestep variation (std mean): {timestep_variation:.6f}")
          print(
            f"  Value range: [{batch_data.min().item():.4f}, {batch_data.max().item():.4f}]"
          )
      # Normalize if needed
      if normalizer is not None:
        batch_normalized = normalizer(batch)
      else:
        batch_normalized = batch

      # Reconstruct
      reconstructed_normalized, latent = model(batch_normalized)

      # Denormalize for visualization
      if normalizer is not None:
        reconstructed = normalizer.denormalize(reconstructed_normalized)
        original = batch  # Already denormalized
      else:
        reconstructed = reconstructed_normalized
        original = batch_normalized

      # Convert to numpy
      original_np = original[0].cpu().numpy()  # (horizon, input_dim)
      reconstructed_np = reconstructed[0].cpu().numpy()  # (horizon, input_dim)

      # Compute metrics
      metrics = compute_reconstruction_metrics(original, reconstructed)
      all_metrics.append(metrics)

      print(f"\nSample {sample_idx + 1}/{num_samples}:")
      print(
        f"  Overall MSE: {metrics['overall_mse']:.6f}, MAE: {metrics['overall_mae']:.6f}"
      )
      for fg in feature_groups:
        print(
          f"  {fg:15s} MSE: {metrics[f'{fg}_mse']:.6f}, MAE: {metrics[f'{fg}_mae']:.6f}"
        )

      # Plot each feature group
      for feature_name in feature_groups:
        feature_range = FEATURE_RANGES[feature_name]
        save_path = output_dir_path / f"sample_{sample_idx + 1:03d}_{feature_name}.png"
        plot_feature_comparison(
          original_np,
          reconstructed_np,
          feature_name,
          feature_range,
          save_path=save_path,
          show_plot=show_plots,
        )

  # Compute average metrics
  print("\n" + "=" * 80)
  print("Average Metrics Across All Samples:")
  print("=" * 80)
  avg_metrics = {}
  for key in all_metrics[0].keys():
    avg_metrics[key] = np.mean([m[key] for m in all_metrics])
    print(f"  {key:20s}: {avg_metrics[key]:.6f}")

  # Save summary
  summary_path = output_dir_path / "summary.txt"
  with open(summary_path, "w") as f:
    f.write("Reconstruction Summary\n")
    f.write("=" * 80 + "\n\n")
    for key, value in avg_metrics.items():
      f.write(f"{key:20s}: {value:.6f}\n")
  print(f"\nSummary saved to: {summary_path}")


def main():
  """Main entry point."""
  parser = argparse.ArgumentParser(description="Visualize reconstructed trajectories")
  parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to model checkpoint (.pt file)",
  )
  parser.add_argument(
    "--num-samples",
    type=int,
    default=5,
    help="Number of samples to visualize (default: 5)",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Output directory for plots (default: checkpoint_dir/visualizations)",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to use (default: cpu)",
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Don't display plots (only save)",
  )
  parser.add_argument(
    "--features",
    type=str,
    nargs="+",
    default=None,
    help="Feature groups to plot (default: all). Options: joint_pos, joint_vel, anchor_pos, anchor_quat, object_pos, object_quat",
  )

  args = parser.parse_args()

  visualize_reconstructions(
    checkpoint_path=args.checkpoint,
    num_samples=args.num_samples,
    output_dir=args.output_dir,
    device=args.device,
    show_plots=not args.no_show,
    feature_groups=args.features,
  )


if __name__ == "__main__":
  main()
