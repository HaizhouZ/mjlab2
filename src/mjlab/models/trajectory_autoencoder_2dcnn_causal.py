"""Causal 2D CNN-based autoencoder for trajectory data.

Treats the trajectory as a 2D image where:
- Height = horizon (time dimension) - CAUSAL: only uses past timesteps
- Width = input_dim (feature dimension) - can use all features

Uses causal convolutions in the temporal dimension to ensure no future information
is used when processing the current timestep.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab.models.initialization import init_autoencoder_weights
from mjlab.models.trajectory_autoencoder_base import TrajectoryAutoencoderBase


class CausalConv2dBlock(nn.Module):
  """Causal 2D Convolutional block with batch norm and activation.

  Causal in the temporal (height) dimension: only uses past timesteps.
  Non-causal in the feature (width) dimension: can use all features.
  """

  def __init__(
    self,
    in_channels: int,
    out_channels: int,
    kernel_size: int | tuple[int, int] = 3,
    stride: int | tuple[int, int] = 1,
    dilation: int | tuple[int, int] = 1,
    use_batch_norm: bool = True,
  ):
    super().__init__()
    if isinstance(kernel_size, int):
      kernel_size = (kernel_size, kernel_size)
    if isinstance(stride, int):
      stride = (stride, stride)
    if isinstance(dilation, int):
      dilation = (dilation, dilation)

    # For causal convolution in temporal dimension (height):
    # - Padding on left (top) = (kernel_height - 1) * dilation_height
    # - Padding on right (bottom) = 0
    # - Padding on feature dimension (width) = kernel_width // 2 (symmetric)
    padding_h = (kernel_size[0] - 1) * dilation[
      0
    ]  # Causal padding for temporal dimension
    padding_w = kernel_size[1] // 2  # Symmetric padding for feature dimension
    padding = (padding_h, padding_w)

    # Store padding for cropping in forward pass
    self.causal_padding_h = padding_h

    self.conv = nn.Conv2d(
      in_channels,
      out_channels,
      kernel_size=kernel_size,
      stride=stride,
      padding=padding,
      dilation=dilation,
    )
    self.bn = nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity()
    self.activation = nn.ReLU(inplace=True)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    x = self.conv(x)
    # Crop the right (bottom) padding that was added by PyTorch's symmetric padding
    # We only want left (top) padding for causality
    # Note: PyTorch's Conv2d with padding applies symmetric padding, so we need to crop
    # the bottom padding to maintain causality
    if self.causal_padding_h > 0:
      # Remove the bottom padding on the temporal dimension
      # Keep only the top padding for causality
      x = x[:, :, : -self.causal_padding_h, :]
    x = self.bn(x)
    x = self.activation(x)
    return x


class TrajectoryAutoencoder2DCNNCausal(TrajectoryAutoencoderBase):
  """Causal 2D CNN-based autoencoder for compressing trajectory data.

  Architecture:
  - Encoder: Stack of causal 2D convolutional blocks that progressively downsample
    in the temporal dimension while maintaining causality
  - Bottleneck: Compress to latent vector
  - Decoder: Stack of causal 2D transposed convolutional blocks that upsample
    back to original dimensions

  Key features:
  - Causal convolutions ensure no future information leaks
  - Useful for online/streaming applications
  - Maintains temporal ordering constraints
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
    """Initialize the causal 2D CNN autoencoder.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        input_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the latent vector
        encoder_channels: List of channel sizes for encoder layers.
                        Default: [32, 64, 128, 256]
        decoder_channels: List of channel sizes for decoder layers.
                        Default: [256, 128, 64, 32] (reversed encoder)
        kernel_size: Kernel size for convolutions. Can be int or (height, width).
                    Default: 3
        use_batch_norm: Whether to use batch normalization. Default: True
    """
    super().__init__(horizon, input_dim, latent_dim)

    if encoder_channels is None:
      encoder_channels = [32, 64, 128, 256]
    if decoder_channels is None:
      decoder_channels = list(reversed(encoder_channels[:-1]))  # Reverse, exclude last

    # Encoder: progressively downsample temporal dimension
    encoder_blocks = []
    in_channels = 1
    current_h = horizon

    for out_channels in encoder_channels:
      # Use stride=2 in temporal dimension for downsampling, stride=1 in feature dimension
      encoder_blocks.append(
        CausalConv2dBlock(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=kernel_size,
          stride=(2, 1),  # Downsample time, keep feature dim
          use_batch_norm=use_batch_norm,
        )
      )
      in_channels = out_channels
      # Calculate output height after causal conv with stride=2
      # For causal conv with padding_top = (kernel_h - 1) and stride=2:
      # After cropping bottom padding: output_h ≈ input_h // 2
      # More precisely: output_h = floor((input_h + padding_top - kernel_h) / stride_h) + 1
      # With padding_top = kernel_h - 1: output_h = floor((input_h - 1) / 2) + 1
      # After cropping, the output is approximately input_h // 2
      current_h = (current_h + 1) // 2

    self.encoder_blocks = nn.ModuleList(encoder_blocks)
    self.encoder_out_channels = encoder_channels[-1]

    # Compute actual encoded dimensions by doing a forward pass with dummy input
    with torch.no_grad():
      dummy_input = torch.zeros(1, 1, horizon, input_dim)
      dummy_output = dummy_input
      for block in self.encoder_blocks:
        dummy_output = block(dummy_output)
      self.encoded_h = dummy_output.shape[2]
      self.encoded_w = dummy_output.shape[3]

    # Bottleneck: compress to latent
    bottleneck_size = self.encoder_out_channels * self.encoded_h * self.encoded_w
    self.bottleneck = nn.Sequential(
      nn.Flatten(),
      nn.Linear(bottleneck_size, latent_dim * 4),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(latent_dim * 4, latent_dim),
    )

    # Decoder: expand from latent and upsample
    self.decoder_fc = nn.Sequential(
      nn.Linear(latent_dim, latent_dim * 4),
      nn.ReLU(),
      nn.Dropout(0.1),
      nn.Linear(
        latent_dim * 4, self.encoder_out_channels * self.encoded_h * self.encoded_w
      ),
    )

    # Decoder blocks: upsample back to original dimensions
    decoder_blocks = []
    in_channels = self.encoder_out_channels

    for out_channels in decoder_channels:
      # Use stride=2 in temporal dimension for upsampling, stride=1 in feature dimension
      decoder_blocks.append(
        nn.Sequential(
          nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size if isinstance(kernel_size, int) else kernel_size,
            stride=(2, 1),  # Upsample time, keep feature dim
            padding=(
              1,
              kernel_size[1] // 2
              if isinstance(kernel_size, tuple)
              else kernel_size // 2,
            ),
            output_padding=(1, 0),  # Adjust for exact upsampling
          ),
          nn.BatchNorm2d(out_channels) if use_batch_norm else nn.Identity(),
          nn.ReLU(inplace=True),
        )
      )
      in_channels = out_channels
      current_h = current_h * 2  # Approximate after stride-2 upsampling

    self.decoder_blocks = nn.ModuleList(decoder_blocks)

    # Final conv to get 1 channel
    if isinstance(kernel_size, int):
      final_padding = (kernel_size // 2, kernel_size // 2)
    else:
      final_padding = (kernel_size[0] // 2, kernel_size[1] // 2)

    self.final_conv = nn.Conv2d(
      in_channels=decoder_channels[-1]
      if decoder_channels
      else self.encoder_out_channels,
      out_channels=1,
      kernel_size=kernel_size if isinstance(kernel_size, int) else kernel_size,
      stride=1,
      padding=final_padding,
    )

    # Initialize weights
    init_autoencoder_weights(self)

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim)

    Returns:
        Latent vector of shape (batch_size, latent_dim)
    """
    # Reshape: (batch_size, horizon, input_dim) -> (batch_size, 1, horizon, input_dim)
    # Treat as grayscale image: channels=1, height=horizon (time), width=input_dim
    x = x.unsqueeze(1)

    # Encoder blocks with causal convolutions
    for block in self.encoder_blocks:
      x = block(x)

    # Flatten and compress to latent
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

    # Expand latent
    x = self.decoder_fc(latent)
    # Reshape: (batch_size, channels * h * w) -> (batch_size, channels, h, w)
    x = x.view(batch_size, self.encoder_out_channels, self.encoded_h, self.encoded_w)

    # Decoder blocks with upsampling
    for i, block in enumerate(self.decoder_blocks):
      x = block(x)
      if i < len(self.decoder_blocks) - 1:
        x = F.relu(x, inplace=False)

    # Final conv to get 1 channel
    x = self.final_conv(x)
    x = F.relu(x, inplace=False)

    # Crop or pad to match original horizon if needed
    if x.shape[2] != self.horizon:
      if x.shape[2] > self.horizon:
        # Crop if too large
        x = x[:, :, : self.horizon, :]
      else:
        # Pad if too small (pad with zeros on the right/bottom)
        pad_size = self.horizon - x.shape[2]
        x = F.pad(x, (0, 0, 0, pad_size))

    # Reshape: (batch_size, 1, horizon, input_dim) -> (batch_size, horizon, input_dim)
    x = x.squeeze(1)

    return x
