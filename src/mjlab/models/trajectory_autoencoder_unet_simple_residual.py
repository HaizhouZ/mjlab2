"""Simple U-Net style autoencoder with residual connections using only linear layers for compressing trajectory data."""

from __future__ import annotations

import torch
import torch.nn as nn

from mjlab.models.initialization import init_autoencoder_weights
from mjlab.models.trajectory_autoencoder_base import TrajectoryAutoencoderBase


class ResidualLinearBlock(nn.Module):
  """Residual block with two linear layers and skip connection."""

  def __init__(
    self,
    dim: int,
    use_batch_norm: bool = True,
    dropout: float = 0.0,
  ):
    super().__init__()
    self.linear1 = nn.Linear(dim, dim)
    self.bn1 = nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity()
    self.linear2 = nn.Linear(dim, dim)
    self.bn2 = nn.BatchNorm1d(dim) if use_batch_norm else nn.Identity()
    self.activation = nn.ReLU(inplace=True)
    self.dropout = nn.Dropout(dropout) if dropout > 0.0 else nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x

    # First linear block
    out = self.linear1(x)
    out = self.bn1(out)
    out = self.activation(out)
    out = self.dropout(out)

    # Second linear block
    out = self.linear2(out)
    out = self.bn2(out)

    # Add residual connection
    out = out + identity
    out = self.activation(out)

    return out


class TrajectoryAutoencoderUNetSimpleResidual(TrajectoryAutoencoderBase):
  """Simple U-Net style autoencoder with residual connections using only linear layers for compressing trajectory data.

  The encoder compresses a trajectory of horizon H with feature dimension D
  to a latent vector. The decoder reconstructs the trajectory from the latent.

  Architecture:
  - Encoder: Linear layers that progressively compress the flattened trajectory with residual connections
  - Bottleneck: Fully connected layers to latent dimension
  - Decoder: Linear layers that progressively expand back to original dimensions with residual connections

  Unlike the standard UNet, this version uses only fully connected (linear) layers
  without any convolutional operations. The trajectory is flattened and processed
  as a single vector. Residual connections are added when dimensions match.

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
    use_residual: bool = True,
  ):
    """Initialize the simple trajectory autoencoder with residual connections.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        input_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the latent vector
        encoder_hidden_dims: List of hidden layer sizes for encoder (default: [512, 256])
        decoder_hidden_dims: List of hidden layer sizes for decoder (default: [256, 512])
        use_batch_norm: Whether to use batch normalization
        dropout: Dropout probability (default: 0.0, no dropout)
        use_residual: Whether to use residual connections (default: True)
    """
    super().__init__(horizon, input_dim, latent_dim)

    # Calculate input size (flattened trajectory)
    input_size = horizon * input_dim

    # Default encoder/decoder hidden dimensions
    if encoder_hidden_dims is None:
      encoder_hidden_dims = [512, 256]
    if decoder_hidden_dims is None:
      decoder_hidden_dims = [256, 512]

    # Encoder: Linear layers that progressively compress with residual connections
    self.encoder_transitions = nn.ModuleList()
    self.encoder_residual_blocks = nn.ModuleList()
    in_dim = input_size
    for hidden_dim in encoder_hidden_dims:
      # Transition layer: change dimensions
      transition = []
      transition.append(nn.Linear(in_dim, hidden_dim))
      if use_batch_norm:
        transition.append(nn.BatchNorm1d(hidden_dim))
      transition.append(nn.ReLU(inplace=True))
      if dropout > 0.0:
        transition.append(nn.Dropout(dropout))
      self.encoder_transitions.append(nn.Sequential(*transition))

      # Residual block: same dimensions
      if use_residual:
        self.encoder_residual_blocks.append(
          ResidualLinearBlock(hidden_dim, use_batch_norm, dropout)
        )
      else:
        self.encoder_residual_blocks.append(nn.Identity())

      in_dim = hidden_dim

    # Bottleneck: Compress to latent dimension
    self.bottleneck = nn.Sequential(
      nn.Linear(in_dim, latent_dim * 2),
      nn.ReLU(inplace=True),
      nn.Linear(latent_dim * 2, latent_dim),
    )

    # Decoder: Expand from latent and progressively reconstruct with residual connections
    self.decoder_transitions = nn.ModuleList()
    self.decoder_residual_blocks = nn.ModuleList()
    in_dim = latent_dim

    # First expand from latent
    first_transition = []
    first_transition.append(nn.Linear(in_dim, latent_dim * 2))
    if use_batch_norm:
      first_transition.append(nn.BatchNorm1d(latent_dim * 2))
    first_transition.append(nn.ReLU(inplace=True))
    if dropout > 0.0:
      first_transition.append(nn.Dropout(dropout))
    self.decoder_transitions.append(nn.Sequential(*first_transition))

    # Residual block for first transition (if dimensions match)
    if use_residual:
      self.decoder_residual_blocks.append(
        ResidualLinearBlock(latent_dim * 2, use_batch_norm, dropout)
      )
    else:
      self.decoder_residual_blocks.append(nn.Identity())

    in_dim = latent_dim * 2

    # Then use decoder hidden dims
    for hidden_dim in decoder_hidden_dims:
      # Transition layer: change dimensions
      transition = []
      transition.append(nn.Linear(in_dim, hidden_dim))
      if use_batch_norm:
        transition.append(nn.BatchNorm1d(hidden_dim))
      transition.append(nn.ReLU(inplace=True))
      if dropout > 0.0:
        transition.append(nn.Dropout(dropout))
      self.decoder_transitions.append(nn.Sequential(*transition))

      # Residual block: same dimensions
      if use_residual:
        self.decoder_residual_blocks.append(
          ResidualLinearBlock(hidden_dim, use_batch_norm, dropout)
        )
      else:
        self.decoder_residual_blocks.append(nn.Identity())

      in_dim = hidden_dim

    # Final layer to reconstruct original size
    self.decoder_final = nn.Linear(in_dim, input_size)

    # Store input size for reshaping
    self.input_size = input_size
    self.use_residual = use_residual

    # Initialize weights with best practices
    init_autoencoder_weights(self)

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim)

    Returns:
        Latent vector of shape (batch_size, latent_dim)
    """
    batch_size = x.shape[0]

    # Flatten: (batch_size, horizon, input_dim) -> (batch_size, horizon * input_dim)
    x = x.view(batch_size, -1)

    # Encoder: progressively compress with residual connections
    for transition, residual_block in zip(
      self.encoder_transitions, self.encoder_residual_blocks, strict=False
    ):
      x = transition(x)
      x = residual_block(x)

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

    # Decoder: progressively expand with residual connections
    x = latent
    for transition, residual_block in zip(
      self.decoder_transitions, self.decoder_residual_blocks, strict=False
    ):
      x = transition(x)
      x = residual_block(x)

    # Final layer to reconstruct original size
    x = self.decoder_final(x)

    # Reshape: (batch_size, horizon * input_dim) -> (batch_size, horizon, input_dim)
    x = x.view(batch_size, self.horizon, self.input_dim)

    return x
