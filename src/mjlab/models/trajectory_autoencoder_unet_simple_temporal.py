"""Simple U-Net style autoencoder with temporal inductive biases for trajectory data.

This variant adds temporal inductive biases to the simple U-Net architecture:
- Temporal positional encodings to help model understand temporal ordering
- Optional temporal convolutions to capture local temporal patterns
- Maintains the efficient linear layer structure of unet_simple
"""

from __future__ import annotations

import torch
import torch.nn as nn

from mjlab.models.initialization import init_autoencoder_weights
from mjlab.models.trajectory_autoencoder_base import TrajectoryAutoencoderBase


class TemporalPositionalEncoding(nn.Module):
  """Learnable positional encoding for temporal sequences.

  Adds learnable embeddings for each temporal position to help the model
  understand temporal ordering and relationships.
  """

  def __init__(self, horizon: int, input_dim: int):
    """Initialize temporal positional encoding.

    Args:
        horizon: Number of timesteps (H)
        input_dim: Feature dimension per timestep (D)
    """
    super().__init__()
    # Learnable positional embeddings: (horizon, input_dim)
    self.pos_embedding = nn.Parameter(torch.randn(horizon, input_dim) * 0.02)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Add positional encoding to input.

    Args:
        x: Input tensor of shape (batch_size, horizon, input_dim)

    Returns:
        Tensor with positional encoding added: (batch_size, horizon, input_dim)
    """
    return x + self.pos_embedding.unsqueeze(0)


class TemporalConvBlock(nn.Module):
  """Temporal convolution block for capturing local temporal patterns.

  Uses 1D convolutions along the temporal dimension to capture local
  temporal dependencies before/after the linear layers.
  """

  def __init__(
    self,
    input_dim: int,
    hidden_dim: int,
    kernel_size: int = 3,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
  ):
    """Initialize temporal convolution block.

    Args:
        input_dim: Input feature dimension
        hidden_dim: Hidden dimension for convolution
        kernel_size: Size of temporal convolution kernel
        use_batch_norm: Whether to use batch normalization
        dropout: Dropout probability
    """
    super().__init__()
    padding = kernel_size // 2  # Same padding

    self.conv = nn.Conv1d(
      input_dim,
      hidden_dim,
      kernel_size=kernel_size,
      padding=padding,
    )
    self.bn = nn.BatchNorm1d(hidden_dim) if use_batch_norm else nn.Identity()
    self.activation = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Apply temporal convolution.

    Args:
        x: Input tensor of shape (batch_size, horizon, input_dim)

    Returns:
        Output tensor of shape (batch_size, horizon, hidden_dim)
    """
    # Transpose: (batch_size, horizon, input_dim) -> (batch_size, input_dim, horizon)
    x = x.transpose(1, 2)
    x = self.conv(x)
    x = self.bn(x)
    x = self.activation(x)
    x = self.dropout(x)
    # Transpose back: (batch_size, hidden_dim, horizon) -> (batch_size, horizon, hidden_dim)
    x = x.transpose(1, 2)
    return x


class TrajectoryAutoencoderUNetSimpleTemporal(TrajectoryAutoencoderBase):
  """Simple U-Net style autoencoder with temporal inductive biases.

  This variant adds temporal inductive biases to help the model better
  understand temporal structure in trajectory data:

  1. Temporal Positional Encodings: Learnable embeddings for each timestep
     to help model understand temporal ordering
  2. Optional Temporal Convolutions: 1D convolutions along temporal dimension
     to capture local temporal patterns before/after linear layers
  3. Maintains efficient linear layer structure from unet_simple

  Architecture:
  - Input: (batch_size, H, D)
  - Temporal positional encoding (optional)
  - Temporal conv preprocessing (optional)
  - Encoder: Linear layers that progressively compress the flattened trajectory
  - Bottleneck: Fully connected layers to latent dimension
  - Decoder: Linear layers that progressively expand back to original dimensions
  - Temporal conv postprocessing (optional)

  Input shape: (batch_size, H, D) where D = 72
  Output shape: (batch_size, H, D) - reconstructed trajectory
  Latent shape: (batch_size, latent_dim)
  """

  def __init__(
    self,
    horizon: int,
    input_dim: int = 72,
    latent_dim: int = 128,
    encoder_hidden_dims: list[int] | None = None,
    decoder_hidden_dims: list[int] | None = None,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
    use_temporal_pos_encoding: bool = True,
    use_temporal_conv: bool = False,
    temporal_conv_dim: int | None = None,
    temporal_conv_kernel_size: int = 3,
  ):
    """Initialize the temporal-aware trajectory autoencoder.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        input_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the latent vector
        encoder_hidden_dims: List of hidden layer sizes for encoder (default: [512, 256])
        decoder_hidden_dims: List of hidden layer sizes for decoder (default: [256, 512])
        use_batch_norm: Whether to use batch normalization
        dropout: Dropout probability (default: 0.0, no dropout)
        use_temporal_pos_encoding: Whether to use temporal positional encodings
        use_temporal_conv: Whether to use temporal convolutions before/after linear layers
        temporal_conv_dim: Hidden dimension for temporal conv (default: input_dim)
        temporal_conv_kernel_size: Kernel size for temporal convolutions
    """
    super().__init__(horizon, input_dim, latent_dim)

    # Temporal positional encoding
    self.use_temporal_pos_encoding = use_temporal_pos_encoding
    if use_temporal_pos_encoding:
      self.temporal_pos_encoding = TemporalPositionalEncoding(horizon, input_dim)
    else:
      self.temporal_pos_encoding = None

    # Temporal convolution preprocessing
    self.use_temporal_conv = use_temporal_conv
    if use_temporal_conv:
      if temporal_conv_dim is None:
        temporal_conv_dim = input_dim
      self.temporal_conv_pre = TemporalConvBlock(
        input_dim,
        temporal_conv_dim,
        kernel_size=temporal_conv_kernel_size,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
      )
      # After temporal conv, we need to project back to input_dim for linear layers
      self.temporal_conv_pre_proj = nn.Linear(temporal_conv_dim, input_dim)
      effective_input_dim = input_dim
    else:
      self.temporal_conv_pre = None
      self.temporal_conv_pre_proj = None
      effective_input_dim = input_dim

    # Calculate input size (flattened trajectory)
    input_size = horizon * effective_input_dim

    # Default encoder/decoder hidden dimensions
    if encoder_hidden_dims is None:
      encoder_hidden_dims = [512, 256]
    if decoder_hidden_dims is None:
      decoder_hidden_dims = [256, 512]

    # Encoder: Linear layers that progressively compress
    encoder_layers = []
    in_dim = input_size
    for hidden_dim in encoder_hidden_dims:
      encoder_layers.append(nn.Linear(in_dim, hidden_dim))
      if use_batch_norm:
        encoder_layers.append(nn.BatchNorm1d(hidden_dim))
      encoder_layers.append(nn.ReLU(inplace=True))
      if dropout > 0.0:
        encoder_layers.append(nn.Dropout(dropout))
      in_dim = hidden_dim

    self.encoder = nn.Sequential(*encoder_layers)

    # Bottleneck: Compress to latent dimension
    self.bottleneck = nn.Sequential(
      nn.Linear(in_dim, latent_dim * 2),
      nn.ReLU(inplace=True),
      nn.Linear(latent_dim * 2, latent_dim),
    )

    # Decoder: Expand from latent and progressively reconstruct
    decoder_layers = []
    in_dim = latent_dim
    # First expand from latent
    decoder_layers.append(nn.Linear(in_dim, latent_dim * 2))
    if use_batch_norm:
      decoder_layers.append(nn.BatchNorm1d(latent_dim * 2))
    decoder_layers.append(nn.ReLU(inplace=True))
    if dropout > 0.0:
      decoder_layers.append(nn.Dropout(dropout))
    in_dim = latent_dim * 2

    # Then use decoder hidden dims
    for hidden_dim in decoder_hidden_dims:
      decoder_layers.append(nn.Linear(in_dim, hidden_dim))
      if use_batch_norm:
        decoder_layers.append(nn.BatchNorm1d(hidden_dim))
      decoder_layers.append(nn.ReLU(inplace=True))
      if dropout > 0.0:
        decoder_layers.append(nn.Dropout(dropout))
      in_dim = hidden_dim

    # Final layer to reconstruct original size
    decoder_layers.append(nn.Linear(in_dim, input_size))

    self.decoder = nn.Sequential(*decoder_layers)

    # Temporal convolution postprocessing
    if use_temporal_conv:
      self.temporal_conv_post_proj = nn.Linear(input_dim, temporal_conv_dim)
      self.temporal_conv_post = TemporalConvBlock(
        temporal_conv_dim,
        input_dim,
        kernel_size=temporal_conv_kernel_size,
        use_batch_norm=use_batch_norm,
        dropout=dropout,
      )
    else:
      self.temporal_conv_post_proj = None
      self.temporal_conv_post = None

    # Store input size for reshaping
    self.input_size = input_size
    self.effective_input_dim = effective_input_dim

    # Initialize weights with best practices
    init_autoencoder_weights(self)

  def _encode_impl(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim), already normalized if normalizer is present

    Returns:
        Latent vector of shape (batch_size, latent_dim)
    """
    batch_size = x.shape[0]

    # Apply temporal positional encoding
    if self.use_temporal_pos_encoding:
      x = self.temporal_pos_encoding(x)

    # Apply temporal convolution preprocessing
    if self.use_temporal_conv:
      x = self.temporal_conv_pre(x)
      x = self.temporal_conv_pre_proj(x)

    # Flatten: (batch_size, horizon, effective_input_dim) -> (batch_size, horizon * effective_input_dim)
    x = x.view(batch_size, -1)

    # Encoder: progressively compress
    x = self.encoder(x)

    # Bottleneck: compress to latent
    latent = self.bottleneck(x)

    return latent

  def decode(self, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent vector to trajectory.

    Args:
        latent: Latent vector of shape (batch_size, latent_dim)

    Returns:
        Reconstructed trajectory of shape (batch_size, horizon, input_dim)
    """
    batch_size = latent.shape[0]

    # Decoder: progressively expand
    x = self.decoder(latent)

    # Reshape: (batch_size, horizon * effective_input_dim) -> (batch_size, horizon, effective_input_dim)
    x = x.view(batch_size, self.horizon, self.effective_input_dim)

    # Apply temporal convolution postprocessing
    if self.use_temporal_conv:
      x = self.temporal_conv_post_proj(x)
      x = self.temporal_conv_post(x)

    return x
