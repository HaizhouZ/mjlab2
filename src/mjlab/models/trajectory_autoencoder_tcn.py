"""TCN (Temporal Convolutional Network) based autoencoder for trajectory data."""

from __future__ import annotations

import torch
import torch.nn as nn

from mjlab.models.initialization import init_autoencoder_weights
from mjlab.models.trajectory_autoencoder_base import TrajectoryAutoencoderBase


class TemporalBlock(nn.Module):
  """Temporal Convolutional Block with dilated convolutions and residual connection."""

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    dilation: int = 1,
    dropout: float = 0.2,
  ):
    super().__init__()
    padding = (kernel_size - 1) * dilation

    self.conv1 = nn.Conv1d(
      in_channels,
      out_channels,
      kernel_size,
      padding=padding,
      dilation=dilation,
    )
    self.chomp1 = Chomp1d(padding)
    self.bn1 = nn.BatchNorm1d(out_channels)
    self.relu1 = nn.ReLU()
    self.dropout1 = nn.Dropout(dropout)

    self.conv2 = nn.Conv1d(
      out_channels,
      out_channels,
      kernel_size,
      padding=padding,
      dilation=dilation,
    )
    self.chomp2 = Chomp1d(padding)
    self.bn2 = nn.BatchNorm1d(out_channels)
    self.relu2 = nn.ReLU()
    self.dropout2 = nn.Dropout(dropout)

    self.downsample = (
      nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    )
    self.relu = nn.ReLU()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    residual = x
    out = self.conv1(x)
    out = self.chomp1(out)
    out = self.bn1(out)
    out = self.relu1(out)
    out = self.dropout1(out)

    out = self.conv2(out)
    out = self.chomp2(out)
    out = self.bn2(out)
    out = self.relu2(out)
    out = self.dropout2(out)

    if self.downsample is not None:
      residual = self.downsample(residual)
    return self.relu(out + residual)


class Chomp1d(nn.Module):
  """Remove padding from the end of temporal sequences."""

  def __init__(self, chomp_size: int):
    super().__init__()
    self.chomp_size = chomp_size

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return x[:, :, : -self.chomp_size].contiguous()


class TCNEncoder(nn.Module):
  """TCN-based encoder."""

  def __init__(
    self,
    input_dim: int,
    num_channels: list[int],
    kernel_size: int = 3,
    dropout: float = 0.2,
  ):
    super().__init__()
    layers = []
    num_levels = len(num_channels)
    for i in range(num_levels):
      dilation_size = 2**i
      in_channels = input_dim if i == 0 else num_channels[i - 1]
      out_channels = num_channels[i]
      layers += [
        TemporalBlock(
          in_channels,
          out_channels,
          kernel_size,
          dilation=dilation_size,
          dropout=dropout,
        )
      ]

    self.network = nn.Sequential(*layers)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.network(x)


class TrajectoryAutoencoderTCN(TrajectoryAutoencoderBase):
  """TCN-based autoencoder for compressing trajectory data.

  Uses dilated convolutions to capture long-range temporal dependencies.
  Good for sequences with varying temporal scales.

  Architecture:
  - Encoder: Stacked temporal blocks with exponentially increasing dilation
  - Bottleneck: Fully connected layers
  - Decoder: Stacked temporal blocks (reverse order)
  """

  def __init__(
    self,
    horizon: int,
    input_dim: int = 72,
    latent_dim: int = 128,
    num_channels: list[int] | None = None,
    kernel_size: int = 3,
    dropout: float = 0.2,
  ):
    """Initialize the TCN autoencoder.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        input_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the latent vector
        num_channels: List of channel sizes for TCN blocks (default: [64, 128, 256])
        kernel_size: Kernel size for temporal convolutions
        dropout: Dropout rate
    """
    super().__init__(horizon, input_dim, latent_dim)

    if num_channels is None:
      num_channels = [64, 128, 256]

    # Encoder: TCN blocks
    self.encoder = TCNEncoder(input_dim, num_channels, kernel_size, dropout)

    # Calculate output size after encoding
    # TCN preserves temporal dimension (with padding)
    encoded_length = horizon
    encoded_channels = num_channels[-1]

    # Bottleneck: Compress to latent
    self.bottleneck = nn.Sequential(
      nn.Flatten(),
      nn.Linear(encoded_channels * encoded_length, latent_dim * 2),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(latent_dim * 2, latent_dim),
    )

    # Decoder: Expand from latent
    self.decoder_fc = nn.Sequential(
      nn.Linear(latent_dim, latent_dim * 2),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(latent_dim * 2, encoded_channels * encoded_length),
    )

    # Decoder: TCN blocks (reverse channel order: 256 -> 128 -> 64 -> 72)
    # Build decoder manually to ensure correct channel progression
    decoder_layers = []
    decoder_channels = list(reversed(num_channels)) + [input_dim]
    for i in range(len(decoder_channels) - 1):
      in_ch = decoder_channels[i] if i == 0 else decoder_channels[i]
      out_ch = decoder_channels[i + 1]
      dilation_size = 2 ** (len(num_channels) - 1 - i)
      decoder_layers.append(
        TemporalBlock(
          in_ch, out_ch, kernel_size, dilation=dilation_size, dropout=dropout
        )
      )
    self.decoder = nn.Sequential(*decoder_layers)

    self.encoded_length = encoded_length
    self.encoded_channels = encoded_channels

    # Initialize weights with best practices
    init_autoencoder_weights(self)

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector."""
    # Reshape: (batch_size, horizon, input_dim) -> (batch_size, input_dim, horizon)
    x = x.transpose(1, 2)

    # TCN encoder
    x = self.encoder(x)

    # Flatten and compress to latent
    latent = self.bottleneck(x)

    return latent

  def decode(self, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent vector to trajectory."""
    batch_size = latent.shape[0]

    # Expand latent
    x = self.decoder_fc(latent)
    # Reshape: (batch_size, channels * length) -> (batch_size, channels, length)
    x = x.view(batch_size, self.encoded_channels, self.encoded_length)

    # TCN decoder
    x = self.decoder(x)

    # Reshape: (batch_size, input_dim, horizon) -> (batch_size, horizon, input_dim)
    x = x.transpose(1, 2)

    return x
