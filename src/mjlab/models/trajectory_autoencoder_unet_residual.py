"""U-Net style autoencoder with residual connections for compressing trajectory data."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab.models.initialization import init_autoencoder_weights
from mjlab.models.trajectory_autoencoder_base import TrajectoryAutoencoderBase


class ResidualBlock(nn.Module):
  """Residual block with two convolutional layers and skip connection."""

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
    self.conv1 = nn.Conv1d(
      in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )
    self.bn1 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
    self.conv2 = nn.Conv1d(
      out_channels, out_channels, kernel_size=kernel_size, stride=1, padding=padding
    )
    self.bn2 = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
    self.activation = nn.ReLU(inplace=True)

    # Skip connection: if input and output channels differ or stride > 1, use projection
    self.use_projection = (in_channels != out_channels) or (stride != 1)
    if self.use_projection:
      self.projection = nn.Conv1d(
        in_channels, out_channels, kernel_size=1, stride=stride, padding=0
      )
      self.projection_bn = (
        nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
      )
    else:
      self.projection = nn.Identity()
      self.projection_bn = nn.Identity()

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    identity = x

    # First conv block
    out = self.conv1(x)
    out = self.bn1(out)
    out = self.activation(out)

    # Second conv block
    out = self.conv2(out)
    out = self.bn2(out)

    # Projection for skip connection if needed
    identity = self.projection(identity)
    identity = self.projection_bn(identity)

    # Add residual connection
    out = out + identity
    out = self.activation(out)

    return out


class UNetResidualBlock(nn.Module):
  """U-Net style block with residual connection for encoder/decoder."""

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int = 3,
    stride: int = 1,
    padding: int = 1,
    use_batch_norm: bool = True,
    use_residual: bool = True,
  ):
    super().__init__()
    self.use_residual = use_residual and (in_channels == out_channels) and (stride == 1)

    if self.use_residual:
      # Use residual block when dimensions match
      self.block = ResidualBlock(
        in_channels, out_channels, kernel_size, stride, padding, use_batch_norm
      )
    else:
      # Use simple conv block when dimensions don't match
      self.conv = nn.Conv1d(
        in_channels,
        out_channels,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
      )
      self.bn = nn.BatchNorm1d(out_channels) if use_batch_norm else nn.Identity()
      self.activation = nn.ReLU(inplace=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    if self.use_residual:
      return self.block(x)
    else:
      x = self.conv(x)
      x = self.bn(x)
      x = self.activation(x)
      return x


class TrajectoryAutoencoderUNetResidual(TrajectoryAutoencoderBase):
  """U-Net style autoencoder with residual connections for compressing trajectory data.

  The encoder compresses a trajectory of horizon H with feature dimension D
  to a latent vector. The decoder reconstructs the trajectory from the latent.

  Architecture:
  - Encoder: Conv1D layers with downsampling (stride=2) and residual connections
  - Bottleneck: Fully connected layers
  - Decoder: Conv1D layers with upsampling (transpose conv) and residual connections

  Residual connections are added when input and output channels match and stride=1.
  When dimensions change, standard convolutions are used.

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
    """Initialize the trajectory autoencoder with residual connections.

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

    # Encoder: Conv1D layers with downsampling and residual connections
    self.encoder_blocks = nn.ModuleList()
    in_ch = input_dim
    for out_ch in encoder_channels:
      # First block: downsampling (stride=2) - no residual
      self.encoder_blocks.append(
        UNetResidualBlock(
          in_ch,
          out_ch,
          kernel_size=3,
          stride=2,
          padding=1,
          use_batch_norm=use_batch_norm,
          use_residual=False,
        )
      )
      # Second block: same dimensions (stride=1) - with residual
      self.encoder_blocks.append(
        UNetResidualBlock(
          out_ch,
          out_ch,
          kernel_size=3,
          stride=1,
          padding=1,
          use_batch_norm=use_batch_norm,
          use_residual=True,
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

    # Decoder: Transpose conv layers with upsampling and residual connections
    self.decoder_blocks = nn.ModuleList()
    in_ch = encoder_channels[-1]
    for out_ch in decoder_channels:
      # First block: upsampling (transpose conv) - no residual
      self.decoder_blocks.append(
        nn.ConvTranspose1d(
          in_ch, out_ch, kernel_size=3, stride=2, padding=1, output_padding=1
        )
      )
      # Second block: same dimensions (stride=1) - with residual
      # We'll apply this after the transpose conv in forward pass
      in_ch = out_ch

    # Store decoder channels for residual blocks
    self.decoder_residual_blocks = nn.ModuleList()
    for ch in decoder_channels:
      self.decoder_residual_blocks.append(
        ResidualBlock(
          ch, ch, kernel_size=3, stride=1, padding=1, use_batch_norm=use_batch_norm
        )
      )

    # Final layer to get back to input_dim
    self.final_conv = nn.Conv1d(
      decoder_channels[-1], input_dim, kernel_size=3, stride=1, padding=1
    )

    # Store encoded length for reshaping
    self.encoded_length = encoded_length
    self.encoder_out_channels = encoder_channels[-1]

    # Initialize weights with best practices
    init_autoencoder_weights(self)

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim)

    Returns:
        Latent vector of shape (batch_size, latent_dim)
    """
    # Reshape: (batch_size, horizon, input_dim) -> (batch_size, input_dim, horizon)
    x = x.transpose(1, 2)

    # Encoder blocks with residual connections
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

    # Decoder blocks with upsampling and residual connections
    for _i, (block, residual_block) in enumerate(
      zip(self.decoder_blocks, self.decoder_residual_blocks, strict=False)
    ):
      # Upsampling with transpose conv
      x = block(x)
      x = F.relu(x, inplace=True)
      # Apply residual block
      x = residual_block(x)

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
