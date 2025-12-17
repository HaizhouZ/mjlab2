"""Simple U-Net style autoencoder using only linear layers for compressing trajectory data."""

from __future__ import annotations

import torch
import torch.nn as nn

from mjlab.models.initialization import init_autoencoder_weights
from mjlab.models.trajectory_autoencoder_base import TrajectoryAutoencoderBase


class TrajectoryAutoencoderUNetSimple(TrajectoryAutoencoderBase):
  """Simple U-Net style autoencoder using only linear layers for compressing trajectory data.

  The encoder compresses a trajectory of horizon H with feature dimension D
  to a latent vector. The decoder reconstructs the trajectory from the latent.

  Architecture:
  - Encoder: Linear layers that progressively compress the flattened trajectory
  - Bottleneck: Fully connected layers to latent dimension
  - Decoder: Linear layers that progressively expand back to original dimensions

  Unlike the standard UNet, this version uses only fully connected (linear) layers
  without any convolutional operations. The trajectory is flattened and processed
  as a single vector.

  Input shape: (batch_size, H, D) where D = 72 (joint_pos + joint_vel + anchor_pos + anchor_quat + object_pos + object_quat)
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
  ):
    """Initialize the simple trajectory autoencoder.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        input_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the latent vector
        encoder_hidden_dims: List of hidden layer sizes for encoder (default: [512, 256])
        decoder_hidden_dims: List of hidden layer sizes for decoder (default: [256, 512])
        use_batch_norm: Whether to use batch normalization
        dropout: Dropout probability (default: 0.0, no dropout)
    """
    super().__init__(horizon, input_dim, latent_dim)

    # Calculate input size (flattened trajectory)
    input_size = horizon * input_dim

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

    # Store input size for reshaping
    self.input_size = input_size

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

    # Flatten: (batch_size, horizon, input_dim) -> (batch_size, horizon * input_dim)
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

    # Reshape: (batch_size, horizon * input_dim) -> (batch_size, horizon, input_dim)
    x = x.view(batch_size, self.horizon, self.input_dim)

    return x
