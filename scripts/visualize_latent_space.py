"""Visualize latent space evolution of trajectories using a trained autoencoder.

This script loads a trained model checkpoint and multiple motion trajectories from wandb,
then visualizes how the latent space representation evolves as a sliding window passes
through each trajectory. Multiple motions are visualized with different colors.

Example usage:
    python scripts/visualize_latent_space.py \\
        --checkpoint logs/trajectory_autoencoder/unet_simple_temporal/best_model.pt \\
        --wandb-entity ataritum-org \\
        --wandb-project wandb-registry-motions \\
        --motion-name-pattern ".*sub8.*" ".*sub10.*" \\
        --stride 1 \\
        --reduction pca
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Qt5Agg", force=False)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.decomposition import PCA  # type: ignore  # noqa: E402
from sklearn.manifold import TSNE  # type: ignore  # noqa: E402

from mjlab.asset_zoo.robots import get_g1_robot_cfg  # noqa: E402
from mjlab.entity import Entity  # noqa: E402
from mjlab.models import (  # noqa: E402
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
from mjlab.models.normalization import TrajectoryNormalizer  # noqa: E402
from mjlab.tasks.tracking.mdp.commands import (  # noqa: E402
  MotionLoader,
  MultiMotionLoader,
)
from mjlab.utils.wandb import (  # noqa: E402
  download_motions_from_wandb,
  get_wandb_motion_cache_dir,
)

WANDB_AVAILABLE = True
SKLEARN_AVAILABLE = True


def load_jit_encoder(
  checkpoint_path: Path,
  device: str = "cpu",
) -> tuple[torch.jit.ScriptModule, dict[str, Any]]:
  """Load JIT-compiled encoder from .jit file.

  Args:
      checkpoint_path: Path to JIT encoder file (.jit)
      device: Device to load encoder on

  Returns:
      Tuple of (jitted_encoder, model_info_dict with horizon, input_dim, latent_dim)
  """
  print(f"Loading JIT encoder from: {checkpoint_path}")

  # Load JIT encoder
  jitted_encoder = torch.jit.load(str(checkpoint_path), map_location=device)
  jitted_encoder.eval()
  print("  JIT encoder loaded successfully")

  # Get model parameters from Hydra config
  checkpoint_dir = checkpoint_path.parent
  config_yaml = checkpoint_dir / ".hydra" / "config.yaml"

  if not config_yaml.exists():
    raise ValueError(
      f"Hydra config not found at {config_yaml}. "
      f"JIT encoder requires config to determine model parameters (horizon, input_dim, latent_dim)."
    )

  import yaml

  with open(config_yaml, "r") as f:
    hydra_cfg = yaml.safe_load(f)

  # Try to infer architecture from config name
  default_architecture = "unet_simple"
  if "hydra" in hydra_cfg and "job" in hydra_cfg["hydra"]:
    config_name = hydra_cfg["hydra"]["job"].get("config_name")
    if config_name:
      default_architecture = config_name
  elif (
    "unet" in str(checkpoint_path)
    or "transformer" in str(checkpoint_path)
    or "tcn" in str(checkpoint_path)
  ):
    path_parts = str(checkpoint_path).split("/")
    for part in path_parts:
      if part in [
        "unet",
        "unet_simple",
        "unet_residual",
        "unet_simple_temporal",
        "transformer",
        "tcn",
        "2dcnn",
        "2dcnn_causal",
        "fsq",
      ]:
        default_architecture = part
        break

  # Extract model parameters from config
  horizon = hydra_cfg.get("horizon")
  input_dim = 72
  latent_dim = hydra_cfg.get("latent_dim")
  #   breakpoint()hydra_cfg
  # Validate that we have all required parameters
  if horizon is None or input_dim is None or latent_dim is None:
    missing = []
    if horizon is None:
      missing.append("horizon")
    if input_dim is None:
      missing.append("input_dim")
    if latent_dim is None:
      missing.append("latent_dim")
    raise ValueError(
      f"Missing required model parameters in config: {missing}. "
      f"Please ensure these are specified in the Hydra config."
    )

  print(f"  Architecture: {default_architecture} (inferred)")
  print(f"  Horizon: {horizon}")
  print(f"  Input dim: {input_dim}")
  print(f"  Latent dim: {latent_dim}")

  model_info = {
    "architecture": default_architecture,
    "horizon": horizon,
    "input_dim": input_dim,
    "latent_dim": latent_dim,
  }

  return jitted_encoder, model_info


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

  # Check if file is a TorchScript file (.jit) - handled separately
  if checkpoint_path.suffix == ".jit":
    raise ValueError(
      "Use load_jit_encoder() for .jit files, or the script will handle it automatically."
    )

  # Load checkpoint with weights_only=False for PyTorch 2.6+ compatibility
  try:
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
  except TypeError:
    # Fallback for older PyTorch versions that don't have weights_only parameter
    checkpoint = torch.load(checkpoint_path, map_location=device)

  # Try to load model parameters from checkpoint, fallback to config
  checkpoint_dir = checkpoint_path.parent
  config_yaml = checkpoint_dir / ".hydra" / "config.yaml"

  # Load config if available
  hydra_cfg = {}
  if config_yaml.exists():
    try:
      import yaml

      with open(config_yaml, "r") as f:
        hydra_cfg = yaml.safe_load(f)
    except Exception as e:
      print(f"  Warning: Could not load config from {config_yaml}: {e}")

  # Try to infer architecture from Hydra config name (e.g., from path like .../unet_simple/...)
  # or from the config itself
  default_architecture = "unet_simple"
  if "hydra" in hydra_cfg and "job" in hydra_cfg["hydra"]:
    config_name = hydra_cfg["hydra"]["job"].get("config_name")
    if config_name:
      default_architecture = config_name
  # Also check if architecture is in the checkpoint directory path
  elif (
    "unet" in str(checkpoint_path)
    or "transformer" in str(checkpoint_path)
    or "tcn" in str(checkpoint_path)
  ):
    path_parts = str(checkpoint_path).split("/")
    for part in path_parts:
      if part in [
        "unet",
        "unet_simple",
        "unet_residual",
        "unet_simple_temporal",
        "transformer",
        "tcn",
        "2dcnn",
        "2dcnn_causal",
        "fsq",
      ]:
        default_architecture = part
        break

  # Extract model parameters from checkpoint or config
  architecture = checkpoint.get(
    "architecture", hydra_cfg.get("architecture", default_architecture)
  )
  horizon = checkpoint.get("horizon", hydra_cfg.get("horizon"))
  input_dim = checkpoint.get("input_dim", hydra_cfg.get("input_dim"))
  latent_dim = checkpoint.get("latent_dim", hydra_cfg.get("latent_dim"))

  # Validate that we have all required parameters
  if horizon is None or input_dim is None or latent_dim is None:
    missing = []
    if horizon is None:
      missing.append("horizon")
    if input_dim is None:
      missing.append("input_dim")
    if latent_dim is None:
      missing.append("latent_dim")
    raise ValueError(
      f"Missing required model parameters in checkpoint and config: {missing}. "
      f"Please ensure the checkpoint contains these fields or they are specified in the Hydra config."
    )

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

  # Get architecture-specific config
  config_key = (
    architecture if not architecture.startswith("2dcnn") else f"_{architecture}"
  )
  arch_cfg = model_cfg.get(config_key, {}) if isinstance(model_cfg, dict) else {}

  def get_model_param(key: str, default: Any = None) -> Any:
    return arch_cfg.get(key, default) if isinstance(arch_cfg, dict) else default

  if architecture == "unet":
    model = TrajectoryAutoencoderUNet(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_channels=get_model_param("encoder_channels"),
      decoder_channels=get_model_param("decoder_channels"),
      use_batch_norm=get_model_param("use_batch_norm", True),
    )
  elif architecture == "unet_residual":
    model = TrajectoryAutoencoderUNetResidual(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_channels=get_model_param("encoder_channels"),
      decoder_channels=get_model_param("decoder_channels"),
      use_batch_norm=get_model_param("use_batch_norm", True),
    )
  elif architecture == "unet_simple":
    model = TrajectoryAutoencoderUNetSimple(
      horizon=horizon,
      input_dim=input_dim,
      latent_dim=latent_dim,
      encoder_hidden_dims=get_model_param("encoder_hidden_dims"),
      decoder_hidden_dims=get_model_param("decoder_hidden_dims"),
      use_batch_norm=get_model_param("use_batch_norm", True),
      dropout=get_model_param("dropout", 0.0),
    )
  elif architecture == "unet_simple_residual":
    model = TrajectoryAutoencoderUNetSimpleResidual(
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
    model = TrajectoryAutoencoderUNetSimpleTemporal(
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

  # Load model weights (this will include normalizer if it's part of the model)
  model.load_state_dict(checkpoint["model_state_dict"])
  model = model.to(device)
  model.eval()

  # Extract normalizer from model if it's attached (new format)
  normalizer = None
  if hasattr(model, "normalizer") and model.normalizer is not None:
    normalizer = model.normalizer
    print("  Normalizer loaded from model (integrated)")
  elif checkpoint.get("normalizer_state_dict") is not None:
    # Fallback: load from separate normalizer_state_dict (old format)
    normalizer = TrajectoryNormalizer().to(device)
    normalizer.load_state_dict(checkpoint["normalizer_state_dict"])
    # Also attach it to model for consistency
    model.normalizer = normalizer
    print("  Normalizer loaded from checkpoint (separate entry)")
  else:
    # Try to load from separate normalizer file
    normalizer_path = checkpoint_path.parent / "normalizer.pt"
    if normalizer_path.exists():
      normalizer = TrajectoryNormalizer().to(device)
      normalizer.load_state_dict(torch.load(normalizer_path, map_location=device))
      # Attach it to model for consistency
      model.normalizer = normalizer
      print(f"  Normalizer loaded from: {normalizer_path}")

  return model, normalizer


def download_motions_from_wandb_project(
  wandb_entity: str,
  wandb_project: str,
  cache_dir: Path | str | None = None,
) -> str:
  """Download motions from wandb project.

  Args:
      wandb_entity: Wandb entity/username (e.g., "ataritum-org")
      wandb_project: Wandb project name (e.g., "wandb-registry-motions")
      cache_dir: Optional cache directory

  Returns:
      Path to the motions directory
  """
  if not WANDB_AVAILABLE:
    raise ImportError(
      "wandb is required to download motions. Install with: pip install wandb"
    )

  print(f"Getting motion directory from wandb: {wandb_entity}/{wandb_project}")
  motion_dir = get_wandb_motion_cache_dir(
    wandb_entity=wandb_entity, wandb_project=wandb_project, cache_dir=cache_dir
  )

  # Check if motion_dir exists, if not, download motions from wandb
  if not Path(motion_dir).exists():
    print(
      f"Motion directory {motion_dir} does not exist, downloading motions from wandb"
    )
    motion_dir = str(
      download_motions_from_wandb(
        wandb_entity=wandb_entity,
        wandb_project=wandb_project,
        cache_dir=cache_dir,
        artifact_type="motions",
      )
    )

  print(f"Using motions from: {motion_dir}")
  return str(motion_dir)


def extract_trajectory_windows_from_motion(
  motion_loader: MotionLoader,
  horizon: int,
  stride: int,
  anchor_body_index: int,
) -> torch.Tensor:
  """Extract trajectory windows from a single motion using sliding window.

  Args:
      motion_loader: MotionLoader instance
      horizon: Window size (number of timesteps)
      stride: Stride for sliding window
      anchor_body_index: Index of anchor body within body_names

  Returns:
      Tensor of shape (num_windows, horizon, 72) containing trajectory windows
  """
  time_step_total = motion_loader.time_step_total
  windows = []

  # Extract windows with stride
  for start_t in range(0, time_step_total - horizon + 1, stride):
    end_t = start_t + horizon

    # Get data for this window
    joint_pos = motion_loader.joint_pos[start_t:end_t].clone()  # (horizon, 29)
    joint_vel = motion_loader.joint_vel[start_t:end_t].clone()  # (horizon, 29)

    # Get anchor body pose
    body_pos_w = motion_loader.body_pos_w[
      start_t:end_t
    ].clone()  # (horizon, num_bodies, 3)
    body_quat_w = motion_loader.body_quat_w[
      start_t:end_t
    ].clone()  # (horizon, num_bodies, 4)
    anchor_pos = body_pos_w[:, anchor_body_index]  # (horizon, 3)
    anchor_quat = body_quat_w[:, anchor_body_index]  # (horizon, 4)

    # Get object pose
    object_pos = motion_loader.object_pos_w[start_t:end_t].clone()  # (horizon, 3)
    object_quat = motion_loader.object_quat_w[start_t:end_t].clone()  # (horizon, 4)

    # Concatenate all features: (horizon, 72)
    trajectory = torch.cat(
      [
        joint_pos,  # (horizon, 29)
        joint_vel,  # (horizon, 29)
        anchor_pos,  # (horizon, 3)
        anchor_quat,  # (horizon, 4)
        object_pos,  # (horizon, 3)
        object_quat,  # (horizon, 4)
      ],
      dim=1,
    )

    windows.append(trajectory)

  return torch.stack(windows)  # (num_windows, horizon, 72)


def extract_trajectory_windows_from_multimotion(
  multi_motion_loader: MultiMotionLoader,
  horizon: int,
  stride: int,
  anchor_body_index: int,
) -> tuple[torch.Tensor, list[int]]:
  """Extract trajectory windows from multiple motions using sliding window.

  Args:
      multi_motion_loader: MultiMotionLoader instance
      horizon: Window size (number of timesteps)
      stride: Stride for sliding window
      anchor_body_index: Index of anchor body within body_names

  Returns:
      Tuple of (windows tensor of shape (num_windows, horizon, 72), motion_indices list)
      motion_indices[i] indicates which motion the i-th window comes from
  """
  all_windows = []
  motion_indices = []

  for motion_idx in range(multi_motion_loader.num_motions):
    motion = multi_motion_loader.motions[motion_idx]
    windows = extract_trajectory_windows_from_motion(
      motion, horizon, stride, anchor_body_index
    )
    all_windows.append(windows)
    motion_indices.extend([motion_idx] * len(windows))

  return torch.cat(all_windows, dim=0), motion_indices


def encode_windows(
  model: TrajectoryAutoencoderBase | torch.jit.ScriptModule,
  windows: torch.Tensor,
  device: str,
  batch_size: int = 32,
) -> np.ndarray:
  """Encode trajectory windows to latent space.

  Args:
      model: Trained autoencoder model or JIT-compiled encoder
      windows: Trajectory windows of shape (num_windows, horizon, 72)
      device: Device to run inference on
      batch_size: Batch size for encoding

  Returns:
      Latent vectors of shape (num_windows, latent_dim)
  """
  if isinstance(model, torch.jit.ScriptModule):
    # JIT encoder - call directly
    model.eval()
  else:
    # Regular model
    model.eval()

  latents = []

  with torch.no_grad():
    for i in range(0, len(windows), batch_size):
      batch = windows[i : i + batch_size].to(device)
      if isinstance(model, torch.jit.ScriptModule):
        # JIT encoder takes input directly
        latent = model(batch)  # (batch_size, latent_dim)
      else:
        # Regular model - use encode method
        latent = model.encode(batch)  # (batch_size, latent_dim)
      latents.append(latent.cpu().numpy())

  return np.concatenate(latents, axis=0)  # (num_windows, latent_dim)


def reduce_to_2d(
  latents: np.ndarray,
  method: str = "pca",
  random_state: int = 42,
) -> np.ndarray:
  """Reduce latent vectors to 2D for visualization.

  Args:
      latents: Latent vectors of shape (num_windows, latent_dim)
      method: Reduction method ("pca" or "tsne")
      random_state: Random state for reproducibility

  Returns:
      2D coordinates of shape (num_windows, 2)
  """
  if not SKLEARN_AVAILABLE:
    raise ImportError(
      "sklearn is required for dimensionality reduction. Install with: pip install scikit-learn"
    )

  if method == "pca":
    if PCA is None:
      raise ImportError("PCA is not available. Install scikit-learn.")
    reducer = PCA(n_components=2, random_state=random_state)
    reduced = reducer.fit_transform(latents)
    print(f"  PCA explained variance ratio: {reducer.explained_variance_ratio_}")
    print(f"  Total explained variance: {reducer.explained_variance_ratio_.sum():.4f}")
    return reduced
  elif method == "tsne":
    if TSNE is None:
      raise ImportError("TSNE is not available. Install scikit-learn.")
    print("  Computing t-SNE (this may take a while)...")
    reducer = TSNE(n_components=2, random_state=random_state, perplexity=30)
    reduced = reducer.fit_transform(latents)
    return reduced
  else:
    raise ValueError(f"Unknown reduction method: {method}. Use 'pca' or 'tsne'")


def visualize_latent_evolution(
  coords_2d: np.ndarray,
  save_path: Path | None = None,
  show_plot: bool = True,
  title: str = "Latent Space Evolution",
  motion_indices: list[int] | None = None,
  motion_names: list[str] | None = None,
  use_3d: bool = False,
) -> None:
  """Visualize the evolution of latent space as trajectory progresses.

  Args:
      coords_2d: 2D coordinates of shape (num_windows, 2)
      save_path: Optional path to save the plot
      show_plot: Whether to display the plot
      title: Plot title
      motion_indices: Optional list indicating which motion each window belongs to
      motion_names: Optional list of motion names for legend
      use_3d: If True, create a 3D plot with time as the Z-axis
  """
  if use_3d:
    fig = plt.figure(figsize=(14, 10))
    ax = fig.add_subplot(111, projection="3d")
  else:
    fig, ax = plt.subplots(figsize=(12, 10))

  num_windows = len(coords_2d)

  # Create time dimension - each motion should start at time=0 and evolve independently
  # Use raw window indices (0, 1, 2, ...) without normalization
  if motion_indices is not None and len(motion_indices) == num_windows:
    # Multiple motions - create time per motion (each starts at 0)
    time_coords = np.zeros(num_windows)
    for motion_idx in set(motion_indices):
      mask = np.array(motion_indices) == motion_idx
      # Use raw window indices: 0, 1, 2, ..., num_windows_in_motion-1
      time_coords[mask] = np.arange(np.sum(mask))
  else:
    # Single motion - use raw window indices
    time_coords = np.arange(num_windows)

  if use_3d:
    # 3D visualization with time as Z-axis
    if motion_indices is not None and len(motion_indices) == num_windows:
      # Multiple motions - color by motion
      num_motions = len(set(motion_indices))
      import matplotlib.cm as cm

      tab20 = cm.get_cmap("tab20")
      colors = tab20(np.linspace(0, 1, num_motions))

      for motion_idx in range(num_motions):
        mask = np.array(motion_indices) == motion_idx
        if not np.any(mask):
          continue

        motion_coords = coords_2d[mask]
        motion_times = time_coords[mask]

        # Plot trajectory path for this motion (no label - don't show in legend)
        ax.plot(
          motion_coords[:, 0],
          motion_coords[:, 1],
          motion_times,
          "o-",
          alpha=0.6,
          linewidth=2,
          markersize=4,
          color=colors[motion_idx],
        )

        # Plot start and end points for this motion
        # Only add legend for first motion to avoid duplicates
        start_label = "Start" if motion_idx == 0 else ""
        end_label = "End" if motion_idx == 0 else ""

        ax.scatter(
          motion_coords[0, 0],
          motion_coords[0, 1],
          motion_times[0],
          s=150,  # type: ignore
          c=[colors[motion_idx]],
          marker="s",
          edgecolors="black",
          linewidths=2,
          zorder=5,
          label=start_label,
        )
        ax.scatter(
          motion_coords[-1, 0],
          motion_coords[-1, 1],
          motion_times[-1],
          s=150,  # type: ignore
          c=[colors[motion_idx]],
          marker="^",
          edgecolors="black",
          linewidths=2,
          zorder=5,
          label=end_label,
        )
    else:
      # Single motion - color by time
      ax.plot(
        coords_2d[:, 0],
        coords_2d[:, 1],
        time_coords,
        "o-",
        alpha=0.6,
        linewidth=2,
        markersize=4,
        color="gray",
        label="Trajectory path",
      )

      # Plot start and end points
      ax.scatter(
        coords_2d[0, 0],
        coords_2d[0, 1],
        time_coords[0],
        s=200,  # type: ignore
        c="green",
        marker="s",
        edgecolors="black",
        linewidths=2,
        label="Start",
        zorder=5,
      )
      ax.scatter(
        coords_2d[-1, 0],
        coords_2d[-1, 1],
        time_coords[-1],
        s=200,  # type: ignore
        c="red",
        marker="s",
        edgecolors="black",
        linewidths=2,
        label="End",
        zorder=5,
      )

    ax.set_xlabel("Latent Dimension 1", fontsize=12)
    ax.set_ylabel("Latent Dimension 2", fontsize=12)
    # Type checker doesn't know this is 3D axes, but we know it is when use_3d=True
    if use_3d:
      ax.set_zlabel("Time (Window Index)", fontsize=12)  # type: ignore
    ax.set_title(title, fontsize=14, fontweight="bold")

    # Add legend only for start/end symbols (not motion names)
    handles, labels = ax.get_legend_handles_labels()
    # Filter to show only "Start" and "End" labels
    start_end_handles = []
    start_end_labels = []
    for handle, label in zip(handles, labels, strict=False):
      if label in ["Start", "End"]:
        start_end_handles.append(handle)
        start_end_labels.append(label)

    if start_end_handles:
      ax.legend(start_end_handles, start_end_labels, loc="best", fontsize=10)

    ax.grid(True, alpha=0.3)

  elif motion_indices is not None and len(motion_indices) == num_windows:
    # Multiple motions - color by motion
    num_motions = len(set(motion_indices))
    import matplotlib.cm as cm

    tab20 = cm.get_cmap("tab20")
    colors = tab20(np.linspace(0, 1, num_motions))

    # Plot each motion separately
    for motion_idx in range(num_motions):
      mask = np.array(motion_indices) == motion_idx
      if not np.any(mask):
        continue

      motion_coords = coords_2d[mask]

      # Plot trajectory path for this motion (no label - don't show in legend)
      ax.plot(
        motion_coords[:, 0],
        motion_coords[:, 1],
        "o-",
        alpha=0.6,
        linewidth=2,
        markersize=4,
        color=colors[motion_idx],
      )

      # Plot start and end points for this motion
      # Only add legend for first motion to avoid duplicates
      start_label = "Start" if motion_idx == 0 else ""
      end_label = "End" if motion_idx == 0 else ""

      ax.scatter(
        motion_coords[0, 0],
        motion_coords[0, 1],
        s=150,
        c=colors[motion_idx],
        marker="s",
        edgecolors="black",
        linewidths=2,
        zorder=5,
        label=start_label,
      )
      ax.scatter(
        motion_coords[-1, 0],
        motion_coords[-1, 1],
        s=150,
        c=colors[motion_idx],
        marker="^",
        edgecolors="black",
        linewidths=2,
        zorder=5,
        label=end_label,
      )

    # Add legend only for start/end symbols (not motion names)
    handles, labels = ax.get_legend_handles_labels()
    # Filter to show only "Start" and "End" labels
    start_end_handles = []
    start_end_labels = []
    for handle, label in zip(handles, labels, strict=False):
      if label in ["Start", "End"]:
        start_end_handles.append(handle)
        start_end_labels.append(label)

    if start_end_handles:
      ax.legend(start_end_handles, start_end_labels, loc="best", fontsize=10)
  else:
    # Single motion - color by time
    # Plot trajectory path
    ax.plot(
      coords_2d[:, 0],
      coords_2d[:, 1],
      "o-",
      alpha=0.6,
      linewidth=2,
      markersize=4,
      color="gray",
      label="Trajectory path",
    )

    # Plot start and end points
    ax.scatter(
      coords_2d[0, 0],
      coords_2d[0, 1],
      s=200,
      c="green",
      marker="s",
      edgecolors="black",
      linewidths=2,
      label="Start",
      zorder=5,
    )
    ax.scatter(
      coords_2d[-1, 0],
      coords_2d[-1, 1],
      s=200,
      c="red",
      marker="s",
      edgecolors="black",
      linewidths=2,
      label="End",
      zorder=5,
    )

    # Color-code points by time
    scatter = ax.scatter(
      coords_2d[:, 0],
      coords_2d[:, 1],
      c=range(num_windows),
      cmap="viridis",
      s=50,
      alpha=0.7,
      edgecolors="black",
      linewidths=0.5,
      zorder=3,
    )

    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax)
    cbar.set_label("Window Index (Time)", rotation=270, labelpad=20)
    # ax.legend(loc="best")

  ax.set_xlabel("Latent Dimension 1", fontsize=12)
  ax.set_ylabel("Latent Dimension 2", fontsize=12)
  ax.set_title(title, fontsize=14, fontweight="bold")
  ax.grid(True, alpha=0.3)

  plt.tight_layout()

  if save_path:
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    print(f"  Saved visualization to: {save_path}")

  if show_plot:
    # Check if we're in an interactive environment
    backend = matplotlib.get_backend()
    is_interactive = matplotlib.is_interactive()

    if not is_interactive:
      print(f"  Note: Matplotlib backend '{backend}' is not interactive.")
      print(f"  Plot saved to {save_path if save_path else 'default location'}")
      print(
        "  To view interactively, ensure you have a display available or use X11 forwarding."
      )
    else:
      print(f"  Displaying plot (backend: {backend})...")

    try:
      plt.show(block=True)  # block=True keeps the window open
    except Exception as e:
      print(f"  Warning: Could not display plot interactively: {e}")
      print(f"  Plot saved to {save_path if save_path else 'default location'}")
      plt.close()
  else:
    plt.close()


def main():
  parser = argparse.ArgumentParser(
    description="Visualize latent space evolution of a trajectory"
  )
  parser.add_argument(
    "--checkpoint",
    type=str,
    required=True,
    help="Path to model checkpoint (.pt file)",
  )
  parser.add_argument(
    "--wandb-entity",
    type=str,
    required=True,
    help="Wandb entity/username (e.g., 'ataritum-org')",
  )
  parser.add_argument(
    "--wandb-project",
    type=str,
    required=True,
    help="Wandb project name (e.g., 'wandb-registry-motions')",
  )
  parser.add_argument(
    "--motion-name-pattern",
    type=str,
    nargs="+",
    default=[".*"],
    help="Regex patterns to match motion names (default: ['.*'] matches all)",
  )
  parser.add_argument(
    "--stride",
    type=int,
    default=1,
    help="Stride for sliding window (default: 1)",
  )
  parser.add_argument(
    "--reduction",
    type=str,
    default="pca",
    choices=["pca", "tsne"],
    help="Dimensionality reduction method (default: pca)",
  )
  parser.add_argument(
    "--device",
    type=str,
    default="cpu",
    help="Device to run inference on (default: cpu)",
  )
  parser.add_argument(
    "--batch-size",
    type=int,
    default=32,
    help="Batch size for encoding (default: 32)",
  )
  parser.add_argument(
    "--output-dir",
    type=str,
    default=None,
    help="Output directory for saving plots (default: checkpoint directory)",
  )
  parser.add_argument(
    "--no-show",
    action="store_true",
    help="Don't display plots (only save)",
  )
  parser.add_argument(
    "--body-names",
    type=str,
    nargs="+",
    default=None,
    help="Body names to use (default: from checkpoint config or default)",
  )
  parser.add_argument(
    "--anchor-body-name",
    type=str,
    default=None,
    help="Anchor body name (default: from checkpoint config or 'torso_link')",
  )

  args = parser.parse_args()

  # Load model or JIT encoder
  checkpoint_path = Path(args.checkpoint)
  use_jit = checkpoint_path.suffix == ".jit"

  if not checkpoint_path.exists():
    # Try to find alternative checkpoint files in the same directory
    checkpoint_dir = checkpoint_path.parent
    if checkpoint_dir.exists():
      # Look for .pt files in the directory
      pt_files = list(checkpoint_dir.glob("*.pt"))
      if pt_files:
        # Prefer best_model.pt if it exists, otherwise use the most recent checkpoint
        best_model = checkpoint_dir / "best_model.pt"
        if best_model in pt_files:
          print(f"Checkpoint not found: {checkpoint_path}")
          print("Found 'best_model.pt' in the same directory. Using it instead.")
          checkpoint_path = best_model
        else:
          # Sort by modification time and use the most recent
          pt_files_sorted = sorted(
            pt_files, key=lambda p: p.stat().st_mtime, reverse=True
          )
          print(f"Checkpoint not found: {checkpoint_path}")
          print(f"Found {len(pt_files)} .pt file(s) in the same directory:")
          for pt_file in pt_files_sorted[:5]:  # Show up to 5 most recent
            print(f"  - {pt_file}")
          if len(pt_files) > 5:
            print(f"  ... and {len(pt_files) - 5} more")
          print(f"\nUsing the most recent checkpoint: {pt_files_sorted[0]}")
          checkpoint_path = pt_files_sorted[0]
      else:
        # Check for .jit files if user specified a .jit file
        if use_jit:
          jit_files = list(checkpoint_dir.glob("*.jit"))
          if jit_files:
            # Use the most recent .jit file
            jit_files_sorted = sorted(
              jit_files, key=lambda p: p.stat().st_mtime, reverse=True
            )
            print(f"Checkpoint not found: {checkpoint_path}")
            print(f"Found {len(jit_files)} .jit file(s) in the same directory:")
            for jit_file in jit_files_sorted[:5]:
              print(f"  - {jit_file}")
            if len(jit_files) > 5:
              print(f"  ... and {len(jit_files) - 5} more")
            print(f"\nUsing the most recent JIT encoder: {jit_files_sorted[0]}")
            checkpoint_path = jit_files_sorted[0]
          else:
            raise FileNotFoundError(f"JIT encoder not found: {checkpoint_path}")
        else:
          # Check for .jit files as fallback
          jit_files = list(checkpoint_dir.glob("*.jit"))
          if jit_files:
            print(f"Checkpoint not found: {checkpoint_path}")
            print(
              f"Found {len(jit_files)} .jit file(s). You can use a .jit file by specifying it directly."
            )
            raise FileNotFoundError(
              f"Checkpoint not found: {checkpoint_path}\n"
              f"Found .jit files but .pt checkpoint file was requested."
            )
          else:
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    else:
      raise FileNotFoundError(f"Checkpoint directory does not exist: {checkpoint_dir}")

  # Update use_jit flag based on actual file
  use_jit = checkpoint_path.suffix == ".jit"

  # Load model or JIT encoder
  if use_jit:
    jitted_encoder, model_info = load_jit_encoder(checkpoint_path, device=args.device)
    model = jitted_encoder  # type: ignore
    normalizer = None  # JIT encoder has normalizer integrated
    horizon = model_info["horizon"]
  else:
    model, normalizer = load_model_from_checkpoint(checkpoint_path, device=args.device)
    # Get model parameters
    horizon = model.horizon  # type: ignore

  # Load training config to get body_names and anchor_body_name
  checkpoint_dir = checkpoint_path.parent
  config_yaml = checkpoint_dir / ".hydra" / "config.yaml"

  body_names = args.body_names
  anchor_body_name = args.anchor_body_name

  if config_yaml.exists() and (body_names is None or anchor_body_name is None):
    try:
      import yaml

      with open(config_yaml, "r") as f:
        hydra_cfg = yaml.safe_load(f)
        if body_names is None:
          body_names = hydra_cfg.get("body_names", [])
        if anchor_body_name is None:
          anchor_body_name = hydra_cfg.get("anchor_body_name", "torso_link")
    except Exception as e:
      print(f"  Warning: Could not load config from {config_yaml}: {e}")

  # Defaults if not found
  if body_names is None or len(body_names) == 0:
    print("  Warning: body_names not found, using default")
    body_names = [
      "torso_link",
      "l_shoulder_link",
      "r_shoulder_link",
      "l_elbow_link",
      "r_elbow_link",
      "l_wrist_link",
      "r_wrist_link",
    ]
  if anchor_body_name is None:
    anchor_body_name = "torso_link"

  print(f"  Body names: {body_names}")
  print(f"  Anchor body: {anchor_body_name}")

  # Get body indexes
  robot_cfg = get_g1_robot_cfg()
  robot = Entity(robot_cfg)
  body_indexes = torch.tensor(
    robot.find_bodies(body_names, preserve_order=True)[0],
    dtype=torch.long,
    device=args.device,
  )
  anchor_body_index = body_names.index(anchor_body_name)

  # Download motions from wandb
  motion_dir = download_motions_from_wandb_project(
    wandb_entity=args.wandb_entity,
    wandb_project=args.wandb_project,
  )

  # Load motions using MultiMotionLoader
  print(f"Loading motions from: {motion_dir}")
  print(f"  Motion name patterns: {args.motion_name_pattern}")
  multi_motion_loader = MultiMotionLoader(
    motion_dir=motion_dir,
    motion_name_pattern=args.motion_name_pattern,
    body_indexes=body_indexes,
    device=args.device,
  )
  print(f"  Loaded {multi_motion_loader.num_motions} motion(s)")
  for i, name in enumerate(multi_motion_loader.traj_names):
    motion = multi_motion_loader.motions[i]
    print(f"    Motion {i}: {name} ({motion.time_step_total} timesteps)")

  # Extract trajectory windows from all motions
  print(f"Extracting trajectory windows (horizon={horizon}, stride={args.stride})...")
  windows, motion_indices = extract_trajectory_windows_from_multimotion(
    multi_motion_loader, horizon, args.stride, anchor_body_index
  )
  print(f"  Extracted {len(windows)} windows from {len(set(motion_indices))} motion(s)")

  # Encode windows to latent space
  print("Encoding windows to latent space...")
  latents = encode_windows(model, windows, args.device, batch_size=args.batch_size)
  print(f"  Latent vectors shape: {latents.shape}")

  # Reduce to 2D
  print(f"Reducing to 2D using {args.reduction.upper()}...")
  coords_2d = reduce_to_2d(latents, method=args.reduction)

  # Visualize
  output_dir = (
    Path(args.output_dir) if args.output_dir else checkpoint_dir / "visualizations"
  )
  output_dir.mkdir(parents=True, exist_ok=True)

  # Create title with trajectory info
  title = (
    f"Latent Space Evolution: {args.wandb_entity}/{args.wandb_project}\n"
    f"({args.reduction.upper()}, sub10_largebox_085_original_sbto-v2-top, {len(set(motion_indices))} trajectories)"
  )

  # Generate both 2D and 3D visualizations
  print("\nGenerating visualizations...")

  # 2D visualization
  output_filename_2d = f"latent_evolution_{args.reduction}_2d.png"
  print("  Creating 2D visualization...")
  visualize_latent_evolution(
    coords_2d,
    save_path=output_dir / output_filename_2d,
    show_plot=not args.no_show,
    title=title,
    motion_indices=motion_indices,
    motion_names=multi_motion_loader.traj_names,
    use_3d=False,
  )

  # 3D visualization
  output_filename_3d = f"latent_evolution_{args.reduction}_3d.png"
  print("  Creating 3D visualization...")
  visualize_latent_evolution(
    coords_2d,
    save_path=output_dir / output_filename_3d,
    show_plot=not args.no_show,
    title=title,
    motion_indices=motion_indices,
    motion_names=multi_motion_loader.traj_names,
    use_3d=True,
  )

  print("\nVisualization complete!")


if __name__ == "__main__":
  main()
