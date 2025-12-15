"""2D CNN-based autoencoder for trajectory data.

Treats the trajectory as a 2D image where:
- Height = horizon (time dimension)
- Width = input_dim (feature dimension)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab.models.initialization import init_autoencoder_weights
from mjlab.models.trajectory_autoencoder_base import TrajectoryAutoencoderBase


class Conv2dBlock(nn.Module):
  """2D Convolutional block with batch norm and activation."""

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    padding: int | tuple[int, int] = 1,
    use_batch_norm: bool = True,
  ):
    super().__init__()
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
      stride = (stride, stride)
    if isinstance(padding, int):
      padding = (padding, padding)

    self.conv = nn.Conv2d(
      in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding
    )
    self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
    self.activation = nn.ReLU(inplace=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    x = self.bn(x)
    x = self.activation(x)
    return x


class TrajectoryAutoencoder2DCNN(TrajectoryAutoencoderBase):
  """2D CNN-based autoencoder for compressing trajectory data.

  Treats trajectory as a 2D image:
  - Height dimension = time (horizon)
  - Width dimension = features (input_dim)
  - Channels = 1 (grayscale image)

  Architecture:
  - Encoder: 2D Conv layers with downsampling (stride=2)
  - Bottleneck: Fully connected layers
  - Decoder: Transpose 2D Conv layers with upsampling
  """

  def __init__(
    self,
    horizon: int,
    input_dim: int = 72,
    latent_dim: int = 128,
    encoder_channels: list[int] | None = None,
    decoder_channels: list[int] | None = None,
    kernel_size: int | tuple[int, int] = 3,
    use_batch_norm: bool = True,
  ):
    """Initialize the 2D CNN autoencoder.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        input_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the latent vector
        encoder_channels: List of channel sizes for encoder (default: [32, 64, 128])
        decoder_channels: List of channel sizes for decoder (default: [128, 64, 32])
        kernel_size: Kernel size for convolutions (int or tuple)
        use_batch_norm: Whether to use batch normalization
    """
    super().__init__(horizon, input_dim, latent_dim)

    if encoder_channels is None:
      encoder_channels = [32, 64, 128]
    if decoder_channels is None:
      decoder_channels = [128, 64, 32]

    # Encoder: 2D Conv layers with downsampling
    self.encoder_blocks = nn.ModuleList()
    in_ch = 1  # Input is treated as grayscale image
    for out_ch in encoder_channels:
      self.encoder_blocks.append(
        Conv2dBlock(
          in_ch,
          out_ch,
          kernel_size=kernel_size,
          stride=2,
          padding=1,
          use_batch_norm=use_batch_norm,
        )
      )
      in_ch = out_ch

    # Calculate size after encoding
    encoded_h = horizon
    encoded_w = input_dim
    for _ in encoder_channels:
      encoded_h = (encoded_h + 1) // 2  # Downsampling
      encoded_w = (encoded_w + 1) // 2

    # Bottleneck: Flatten and compress to latent
    self.bottleneck = nn.Sequential(
      nn.Flatten(),
      nn.Linear(encoder_channels[-1] * encoded_h * encoded_w, latent_dim * 2),
      nn.ReLU(inplace=True),
      nn.Linear(latent_dim * 2, latent_dim),
    )

    # Decoder: Expand latent
    self.decoder_fc = nn.Sequential(
      nn.Linear(latent_dim, latent_dim * 2),
      nn.ReLU(inplace=True),
      nn.Linear(latent_dim * 2, encoder_channels[-1] * encoded_h * encoded_w),
    )

    # Decoder: Transpose 2D Conv layers with upsampling
    self.decoder_blocks = nn.ModuleList()
    in_ch = encoder_channels[-1]
    for out_ch in decoder_channels:
      self.decoder_blocks.append(
        nn.ConvTranspose2d(
          in_ch,
          out_ch,
          kernel_size=kernel_size if isinstance(kernel_size, int) else kernel_size,
          stride=2,
          padding=1,
          output_padding=1,
        )
      )
      in_ch = out_ch

    # Final layer to get back to 1 channel
    self.final_conv = nn.Conv2d(
      decoder_channels[-1],
      1,
      kernel_size=kernel_size if isinstance(kernel_size, int) else kernel_size,
      stride=1,
      padding=1,
    )

    self.encoded_h = encoded_h
    self.encoded_w = encoded_w
    self.encoder_out_channels = encoder_channels[-1]

    # Initialize weights with best practices
    init_autoencoder_weights(self)

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector."""
    # Reshape: (batch_size, horizon, input_dim) -> (batch_size, 1, horizon, input_dim)
    # Treat as grayscale image: channels=1, height=horizon, width=input_dim
    x = x.unsqueeze(1)

    # Encoder blocks
    for block in self.encoder_blocks:
      x = block(x)

    # Flatten and compress to latent
    latent = self.bottleneck(x)

    return latent

  def decode(self, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent vector to trajectory."""
    batch_size = latent.shape[0]

    # Expand latent
    x = self.decoder_fc(latent)
    # Reshape: (batch_size, channels * h * w) -> (batch_size, channels, h, w)
    x = x.view(batch_size, self.encoder_out_channels, self.encoded_h, self.encoded_w)

    # Decoder blocks with upsampling
    for i, block in enumerate(self.decoder_blocks):
      x = block(x)
      if i < len(self.decoder_blocks) - 1:
        x = F.relu(x, inplace=True)

    # Final conv to get 1 channel
    x = self.final_conv(x)
    x = F.relu(x, inplace=True)

    # Crop to match exact input dimensions (horizon, input_dim)
    # x shape: (batch_size, 1, H, W) where H and W may be slightly larger than horizon and input_dim
    _, _, current_h, current_w = x.shape
    if current_h != self.horizon or current_w != self.input_dim:
      # Calculate crop offsets (center crop)
      crop_h = (current_h - self.horizon) // 2
      crop_w = (current_w - self.input_dim) // 2
      x = x[:, :, crop_h : crop_h + self.horizon, crop_w : crop_w + self.input_dim]

    # Reshape: (batch_size, 1, horizon, input_dim) -> (batch_size, horizon, input_dim)
    x = x.squeeze(1)

    return x
