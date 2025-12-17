"""Finite Scalar Quantization (FSQ) autoencoder for compressing trajectory data.

This implementation uses FSQ from vector-quantize-pytorch to quantize the latent
representation, providing a discrete codebook-free quantization approach.
Uses only MLP (linear) layers, no convolutions.
"""

from __future__ import annotations

import torch
import torch.nn as nn

try:
  from vector_quantize_pytorch import FSQ

  FSQ_AVAILABLE = True
except ImportError:
  FSQ_AVAILABLE = False
  FSQ = None  # type: ignore

from mjlab.models.initialization import init_autoencoder_weights
from mjlab.models.trajectory_autoencoder_base import TrajectoryAutoencoderBase


class TrajectoryAutoencoderFSQ(TrajectoryAutoencoderBase):
  """Finite Scalar Quantization (FSQ) autoencoder for trajectory data.

  This implementation uses FSQ to quantize the latent representation, providing
  a discrete codebook-free quantization approach. The architecture consists of:

  1. Encoder: MLP layers -> Project each timestep to fsq_dim -> FSQ quantization -> Pool
  2. Decoder: Expand -> Project from fsq_dim -> MLP layers -> Reconstruct

  FSQ quantizes each dimension independently to discrete levels, eliminating the
  need for codebook maintenance and commitment losses. Uses only MLP (linear) layers.

  Input shape: (batch_size, H, D) where D = 72
  Output shape: (batch_size, H, D) - reconstructed trajectory
  Latent shape: (batch_size, fsq_dim) - quantized latent codes
  """

  def __init__(
    self,
    horizon: int,
    input_dim: int = 72,
    latent_dim: int = 128,
    hidden_dim: int = 256,
    fsq_dim: int = 8,
    fsq_levels: list[int] | None = None,
    num_temporal_layers: int = 2,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
  ):
    """Initialize the FSQ trajectory autoencoder.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        input_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the continuous latent before FSQ (unused but kept for compatibility)
        hidden_dim: Hidden dimension for MLP layers
        fsq_dim: Dimension of the FSQ quantized latent
        fsq_levels: Quantization levels for each FSQ dimension (default: [4]*fsq_dim)
        num_temporal_layers: Number of MLP layers for processing each timestep
        use_batch_norm: Whether to use batch normalization
        dropout: Dropout probability (default: 0.0, no dropout)
    """
    if not FSQ_AVAILABLE:
      raise ImportError(
        "vector-quantize-pytorch is required for FSQ. Install it with: pip install vector-quantize-pytorch"
      )

    super().__init__(horizon, input_dim, latent_dim)

    # Set fsq_dim as the actual latent dimension
    self.fsq_dim = fsq_dim
    self.hidden_dim = hidden_dim

    # Default FSQ levels: [4, 4, 4, 4, 4, 4, 4, 4] for 8 dimensions
    if fsq_levels is None:
      fsq_levels = [4] * fsq_dim
    elif len(fsq_levels) != fsq_dim:
      raise ValueError(
        f"fsq_levels length ({len(fsq_levels)}) must match fsq_dim ({fsq_dim})"
      )

    # Encoder: MLP layers for processing each timestep
    # Process each timestep independently with shared MLP layers
    encoder_feature_dim = 128
    self.encoder_mlp_layers = nn.ModuleList()
    in_dim = input_dim
    for _ in range(num_temporal_layers):
      layer = nn.Sequential()
      layer.add_module("linear", nn.Linear(in_dim, hidden_dim))
      if use_batch_norm:
        layer.add_module("bn", nn.BatchNorm1d(hidden_dim))
      layer.add_module("activation", nn.ReLU(inplace=True))
      if dropout > 0.0:
        layer.add_module("dropout", nn.Dropout(dropout))
      self.encoder_mlp_layers.append(layer)
      in_dim = hidden_dim

    # Project from MLP output to a feature dimension for per-timestep encoding
    # We'll encode each timestep separately to preserve temporal structure
    self.encoder_feature_proj = nn.Linear(hidden_dim, encoder_feature_dim)

    # Project each timestep to FSQ dimension
    self.encoder_fsq_proj = nn.Linear(encoder_feature_dim, fsq_dim)

    # Pooling layer to aggregate quantized timesteps into single latent vector
    # This maintains the interface requirement of (batch, latent_dim)
    self.encoder_pool = nn.Linear(horizon * fsq_dim, fsq_dim)

    # FSQ quantizer
    if FSQ is None:
      raise ImportError(
        "vector-quantize-pytorch is required for FSQ. Install it with: pip install vector-quantize-pytorch"
      )
    self.fsq = FSQ(levels=fsq_levels)

    # Decoder: Expand single latent vector back to per-timestep representation
    # We'll expand (batch, fsq_dim) to (batch, horizon, fsq_dim)
    self.decoder_expand = nn.Linear(fsq_dim, horizon * fsq_dim)

    # Project from FSQ dimension back to feature dimension
    self.decoder_fsq_proj = nn.Linear(fsq_dim, encoder_feature_dim)

    # Decoder: MLP layers for processing each timestep
    # Process each timestep independently with shared MLP layers
    self.decoder_mlp_layers = nn.ModuleList()
    in_dim = encoder_feature_dim
    for _ in range(num_temporal_layers):
      layer = nn.Sequential()
      layer.add_module("linear", nn.Linear(in_dim, hidden_dim))
      if use_batch_norm:
        layer.add_module("bn", nn.BatchNorm1d(hidden_dim))
      layer.add_module("activation", nn.ReLU(inplace=True))
      if dropout > 0.0:
        layer.add_module("dropout", nn.Dropout(dropout))
      self.decoder_mlp_layers.append(layer)
      in_dim = hidden_dim

    # Final projection back to input_dim
    self.decoder_final_proj = nn.Linear(hidden_dim, input_dim)

    # Store dimensions
    self.encoder_feature_dim = encoder_feature_dim

    # Initialize weights with best practices
    init_autoencoder_weights(self)

  def _encode_impl(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to quantized latent vector.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim), already normalized if normalizer is present

    Returns:
        Quantized latent vector of shape (batch_size, fsq_dim)
    """
    # Apply MLP layers to each timestep
    # x: (batch_size, horizon, input_dim)
    batch_size, horizon, _ = x.shape

    # Reshape to process all timesteps at once: (batch_size * horizon, input_dim)
    x = x.view(batch_size * horizon, -1)

    # Apply MLP layers
    for mlp_layer in self.encoder_mlp_layers:
      x = mlp_layer(x)
    # x: (batch_size * horizon, hidden_dim)

    # Reshape back: (batch_size * horizon, hidden_dim) -> (batch_size, horizon, hidden_dim)
    x = x.view(batch_size, horizon, -1)

    # Project each timestep to feature dimension
    x = self.encoder_feature_proj(x)  # (batch_size, horizon, encoder_feature_dim)

    # Project each timestep to FSQ dimension
    x = self.encoder_fsq_proj(x)  # (batch_size, horizon, fsq_dim)

    # Quantize using FSQ along the temporal dimension
    # FSQ expects (batch_size, sequence_length, dim)
    quantized, _ = self.fsq(x)  # (batch_size, horizon, fsq_dim)

    # Pool quantized timesteps into single latent vector
    # Flatten temporal dimension: (batch_size, horizon, fsq_dim) -> (batch_size, horizon * fsq_dim)
    quantized_flat = quantized.view(
      quantized.shape[0], -1
    )  # (batch_size, horizon * fsq_dim)
    # Pool to single vector: (batch_size, horizon * fsq_dim) -> (batch_size, fsq_dim)
    latent = self.encoder_pool(quantized_flat)  # (batch_size, fsq_dim)

    return latent

  def decode(self, latent: torch.Tensor) -> torch.Tensor:
    """Decode quantized latent vector to trajectory.

    Args:
        latent: Quantized latent vector of shape (batch_size, fsq_dim)

    Returns:
        Reconstructed trajectory of shape (batch_size, horizon, input_dim)
    """
    batch_size = latent.shape[0]

    # Expand single latent vector back to per-timestep representation
    # (batch_size, fsq_dim) -> (batch_size, horizon * fsq_dim)
    x = self.decoder_expand(latent)
    # Reshape to per-timestep: (batch_size, horizon * fsq_dim) -> (batch_size, horizon, fsq_dim)
    x = x.view(batch_size, self.horizon, self.fsq_dim)

    # Project from FSQ dimension back to feature dimension
    x = self.decoder_fsq_proj(x)  # (batch_size, horizon, encoder_feature_dim)

    # Apply MLP layers to each timestep
    batch_size, horizon, _ = x.shape

    # Reshape to process all timesteps at once: (batch_size * horizon, encoder_feature_dim)
    x = x.view(batch_size * horizon, -1)

    # Apply MLP layers
    for mlp_layer in self.decoder_mlp_layers:
      x = mlp_layer(x)
    # x: (batch_size * horizon, hidden_dim)

    # Reshape back: (batch_size * horizon, hidden_dim) -> (batch_size, horizon, hidden_dim)
    x = x.view(batch_size, horizon, -1)

    # Final projection back to input_dim
    x = self.decoder_final_proj(x)  # (batch_size, horizon, input_dim)

    return x
