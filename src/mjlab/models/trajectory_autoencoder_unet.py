"""U-Net style autoencoder for compressing trajectory data."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab.models.initialization import init_autoencoder_weights
from mjlab.models.trajectory_autoencoder_base import TrajectoryAutoencoderBase


class UNetBlock(nn.Module):
  """U-Net style encoder/decoder block with skip connections."""

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    use_batch_norm: bool = True,
  ):
    super().__init__()
    self.conv = nn.Conv1d(
      in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )
    self.bn = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
    self.activation = nn.ReLU(inplace=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    x = self.bn(x)
    x = self.activation(x)
    return x


class TrajectoryAutoencoderUNet(TrajectoryAutoencoderBase):
  """U-Net style autoencoder for compressing trajectory data.

  The encoder compresses a trajectory of horizon H with feature dimension D
  to a latent vector. The decoder reconstructs the trajectory from the latent.

  Architecture:
  - Encoder: Conv1D layers with downsampling (stride=2)
  - Bottleneck: Fully connected layers
  - Decoder: Conv1D layers with upsampling (transpose conv or interpolation)

  Input shape: (batch_size, H, D) where D = 72 (joint_pos + joint_vel + anchor_pos + anchor_quat + object_pos + object_quat)
  Output shape: (batch_size, H, D) - reconstructed trajectory
  Latent shape: (batch_size, latent_dim)
  """

  def __init__(
    self,
    horizon: int,
    input_dim: int = 72,
    latent_dim: int = 128,
    encoder_channels: list[int] | None = None,
    decoder_channels: list[int] | None = None,
    use_batch_norm: bool = True,
  ):
    """Initialize the trajectory autoencoder.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        input_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the latent vector
        encoder_channels: List of channel sizes for encoder (default: [64, 128, 256])
        decoder_channels: List of channel sizes for decoder (default: [256, 128, 64])
        use_batch_norm: Whether to use batch normalization
    """
    super().__init__(horizon, input_dim, latent_dim)

    # Default encoder/decoder channels
    if encoder_channels is None:
      encoder_channels = [64, 128, 256]
    if decoder_channels is None:
      decoder_channels = [256, 128, 64]

    # Encoder: Conv1D layers with downsampling
    self.encoder_blocks = nn.ModuleList()
    in_ch = input_dim
    for out_ch in encoder_channels:
      self.encoder_blocks.append(
        UNetBlock(
          in_ch,
          out_ch,
          kernel_size=3,
          stride=2,
          padding=1,
          use_batch_norm=use_batch_norm,
        )
      )
      in_ch = out_ch

    # Calculate the size after encoding
    # Each stride=2 reduces the temporal dimension by half
    encoded_length = horizon
    for _ in encoder_channels:
      encoded_length = (encoded_length + 1) // 2  # Account for stride=2

    # Bottleneck: Flatten and compress to latent
    self.bottleneck_conv = nn.Conv1d(
      encoder_channels[-1], encoder_channels[-1], kernel_size=1, stride=1, padding=0
    )
    self.bottleneck_fc = nn.Sequential(
      nn.Linear(encoder_channels[-1] * encoded_length, latent_dim * 2),
      nn.ReLU(inplace=True),
      nn.Linear(latent_dim * 2, latent_dim),
    )

    # Decoder: Expand latent and upsample
    self.decoder_fc = nn.Sequential(
      nn.Linear(latent_dim, latent_dim * 2),
      nn.ReLU(inplace=True),
      nn.Linear(latent_dim * 2, encoder_channels[-1] * encoded_length),
    )

    # Decoder: Transpose conv layers with upsampling
    self.decoder_blocks = nn.ModuleList()
    in_ch = encoder_channels[-1]
    for out_ch in decoder_channels:
      self.decoder_blocks.append(
        nn.ConvTranspose1d(
          in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
        )
      )
      in_ch = out_ch

    # Final layer to get back to input_dim
    self.final_conv = nn.Conv1d(
      decoder_channels[-1], input_dim, kernel_size=3, stride=1, padding=1
    )

    # Store encoded length for reshaping
    self.encoded_length = encoded_length
    self.encoder_out_channels = encoder_channels[-1]

    # Initialize weights with best practices
    init_autoencoder_weights(self)

  def _encode_impl(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim), already normalized if normalizer is present

    Returns:
        Latent vector of shape (batch_size, latent_dim)
    """
    # Reshape: (batch_size, horizon, input_dim) -> (batch_size, input_dim, horizon)
    x = x.transpose(1, 2)

    # Encoder blocks
    for block in self.encoder_blocks:
      x = block(x)

    # Bottleneck
    x = self.bottleneck_conv(x)
    # Flatten: (batch_size, channels, length) -> (batch_size, channels * length)
    batch_size = x.shape[0]
    x = x.view(batch_size, -1)
    # Compress to latent
    latent = self.bottleneck_fc(x)

    return latent

  def decode(self, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent vector to trajectory.

    Args:
        latent: Latent vector of shape (batch_size, latent_dim)

    Returns:
        Reconstructed trajectory of shape (batch_size, horizon, input_dim)
    """
    batch_size = latent.shape[0]

    # Expand latent
    x = self.decoder_fc(latent)
    # Reshape: (batch_size, channels * length) -> (batch_size, channels, length)
    x = x.view(batch_size, self.encoder_out_channels, self.encoded_length)

    # Decoder blocks with upsampling
    for i, block in enumerate(self.decoder_blocks):
      x = block(x)
      # Apply activation and batch norm (if needed)
      if i < len(self.decoder_blocks) - 1:
        x = F.relu(x, inplace=True)

    # Final conv to get input_dim
    x = self.final_conv(x)
    x = F.relu(x, inplace=True)

    # Crop to match exact input horizon dimension
    # x shape: (batch_size, input_dim, current_length) where current_length may be larger than horizon
    _, _, current_length = x.shape
    if current_length != self.horizon:
      # Calculate crop offset (center crop)
      crop_start = (current_length - self.horizon) // 2
      x = x[:, :, crop_start : crop_start + self.horizon]

    # Reshape: (batch_size, input_dim, horizon) -> (batch_size, horizon, input_dim)
    x = x.transpose(1, 2)

    return x
