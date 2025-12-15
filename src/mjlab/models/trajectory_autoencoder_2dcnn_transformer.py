"""2D CNN + Transformer hybrid autoencoder for trajectory data.

Treats the trajectory as a 2D image (horizon x feature_dim), applies 2D convolutions
to extract features, concatenates with the original image, then uses transformer
layers for encoding/decoding.
"""

from __future__ import annotations

import math

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


class PositionalEncoding(nn.Module):
  """Positional encoding for transformer."""

  def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len, dtype=torch.float32).unsqueeze(1)
    div_term = torch.exp(
      torch.arange(0, d_model, 2, dtype=torch.float32) * (-math.log(10000.0) / d_model)
    )
    pe = torch.zeros(max_len, 1, d_model, dtype=torch.float32)
    pe[:, 0, 0::2] = torch.sin(position * div_term)  # type: ignore
    pe[:, 0, 1::2] = torch.cos(position * div_term)  # type: ignore
    self.register_buffer("pe", pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
    """
    x = x + self.pe[: x.size(0)]  # type: ignore[index]
    return self.dropout(x)


class TransformerEncoderBlock(nn.Module):
  """Transformer encoder block with self-attention."""

  def __init__(
    self,
    d_model: int,
    nhead: int = 8,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: str = "relu",
  ):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(
      d_model, nhead, dropout=dropout, batch_first=False
    )

    # Feedforward network
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)

    if activation == "relu":
      self.activation = F.relu
    elif activation == "gelu":
      self.activation = F.gelu
    else:
      self.activation = F.relu

  def forward(
    self, src: torch.Tensor, src_mask: torch.Tensor | None = None
  ) -> torch.Tensor:
    # Self-attention
    src2 = self.self_attn(src, src, src, attn_mask=src_mask)[0]
    src = src + self.dropout1(src2)
    src = self.norm1(src)

    # Feedforward
    src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
    src = src + self.dropout2(src2)
    src = self.norm2(src)

    return src


class TransformerDecoderBlock(nn.Module):
  """Transformer decoder block with self-attention and cross-attention."""

  def __init__(
    self,
    d_model: int,
    nhead: int = 8,
    dim_feedforward: int = 2048,
    dropout: float = 0.1,
    activation: str = "relu",
  ):
    super().__init__()
    self.self_attn = nn.MultiheadAttention(
      d_model, nhead, dropout=dropout, batch_first=False
    )
    self.multihead_attn = nn.MultiheadAttention(
      d_model, nhead, dropout=dropout, batch_first=False
    )

    # Feedforward network
    self.linear1 = nn.Linear(d_model, dim_feedforward)
    self.dropout = nn.Dropout(dropout)
    self.linear2 = nn.Linear(dim_feedforward, d_model)

    self.norm1 = nn.LayerNorm(d_model)
    self.norm2 = nn.LayerNorm(d_model)
    self.norm3 = nn.LayerNorm(d_model)
    self.dropout1 = nn.Dropout(dropout)
    self.dropout2 = nn.Dropout(dropout)
    self.dropout3 = nn.Dropout(dropout)

    if activation == "relu":
      self.activation = F.relu
    elif activation == "gelu":
      self.activation = F.gelu
    else:
      self.activation = F.relu

  def forward(
    self,
    tgt: torch.Tensor,
    memory: torch.Tensor,
    tgt_mask: torch.Tensor | None = None,
    memory_mask: torch.Tensor | None = None,
  ) -> torch.Tensor:
    # Self-attention
    tgt2 = self.self_attn(tgt, tgt, tgt, attn_mask=tgt_mask)[0]
    tgt = tgt + self.dropout1(tgt2)
    tgt = self.norm1(tgt)

    # Cross-attention
    tgt2 = self.multihead_attn(tgt, memory, memory, attn_mask=memory_mask)[0]
    tgt = tgt + self.dropout2(tgt2)
    tgt = self.norm2(tgt)

    # Feedforward
    tgt2 = self.linear2(self.dropout(self.activation(self.linear1(tgt))))
    tgt = tgt + self.dropout3(tgt2)
    tgt = self.norm3(tgt)

    return tgt


class TrajectoryAutoencoder2DCNNTransformer(TrajectoryAutoencoderBase):
  """2D CNN + Transformer hybrid autoencoder for compressing trajectory data.

  Architecture:
  1. Treat input as 2D image: (batch, horizon, feature_dim) -> (batch, 1, horizon, feature_dim)
  2. Apply 2D CNN layers to extract features
  3. Concatenate CNN features with original input (after projection)
  4. Reshape to sequence: (batch, horizon, d_model)
  5. Apply transformer encoder layers
  6. Compress to latent vector
  7. Decode: expand latent, apply transformer decoder, project back to feature_dim

  The key idea is that 2D CNNs capture local spatial patterns in the trajectory
  image, while transformers capture long-range temporal dependencies.
  """

  def __init__(
    self,
    horizon: int,
    feature_dim: int = 72,
    latent_dim: int = 128,
    cnn_channels: list[int] | None = None,
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    activation: str = "relu",
    use_residual_connection: bool = True,
  ):
    """Initialize the 2D CNN + Transformer autoencoder.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        feature_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the latent vector
        cnn_channels: List of channel sizes for 2D CNN layers (default: [32, 64, 128])
        d_model: Dimension of transformer model (default: 256)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 4)
        num_decoder_layers: Number of decoder layers (default: 4)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        dropout: Dropout rate (default: 0.1)
        activation: Activation function ('relu' or 'gelu', default: 'relu')
        use_residual_connection: Whether to concatenate CNN features with original input
    """
    super().__init__(horizon, feature_dim, latent_dim)

    if cnn_channels is None:
      cnn_channels = [32, 64, 128]

    self.d_model = d_model
    self.use_residual_connection = use_residual_connection

    # 2D CNN Encoder: Extract features from 2D image
    # Input: (batch, 1, horizon, feature_dim)
    cnn_layers = []
    in_channels = 1
    for out_channels in cnn_channels:
      cnn_layers.append(
        Conv2dBlock(
          in_channels=in_channels,
          out_channels=out_channels,
          kernel_size=3,
          stride=1,
          padding=1,
          use_batch_norm=True,
        )
      )
      in_channels = out_channels

    self.cnn_encoder = nn.Sequential(*cnn_layers)
    self.cnn_output_channels = cnn_channels[-1]

    # Project concatenated features (CNN + original) to d_model
    # After concatenation: (batch, 1 + cnn_output_channels, horizon, feature_dim) if use_residual_connection
    #                      (batch, cnn_output_channels, horizon, feature_dim) otherwise
    if use_residual_connection:
      projection_input_size = 1 + self.cnn_output_channels
    else:
      projection_input_size = self.cnn_output_channels

    # Projection layer: takes concatenated channels and projects to d_model
    self.cnn_projection = nn.Linear(projection_input_size, d_model)

    # Positional encoding for transformer
    self.pos_encoder = PositionalEncoding(d_model, max_len=horizon * 2, dropout=dropout)
    self.pos_decoder = PositionalEncoding(d_model, max_len=horizon * 2, dropout=dropout)

    # Transformer encoder layers
    encoder_layers = []
    for _ in range(num_encoder_layers):
      encoder_layers.append(
        TransformerEncoderBlock(
          d_model=d_model,
          nhead=nhead,
          dim_feedforward=dim_feedforward,
          dropout=dropout,
          activation=activation,
        )
      )
    self.transformer_encoder = nn.ModuleList(encoder_layers)

    # Bottleneck: compress sequence to latent
    # Use global average pooling + MLP
    self.bottleneck_pool = nn.AdaptiveAvgPool1d(
      1
    )  # (batch, d_model, horizon) -> (batch, d_model, 1)
    self.bottleneck_mlp = nn.Sequential(
      nn.Linear(d_model, latent_dim * 2),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(latent_dim * 2, latent_dim),
    )

    # Decoder: expand latent to sequence
    self.decoder_expand = nn.Sequential(
      nn.Linear(latent_dim, latent_dim * 2),
      nn.ReLU(),
      nn.Dropout(dropout),
      nn.Linear(latent_dim * 2, d_model * horizon),
    )

    # Transformer decoder layers
    decoder_layers = []
    for _ in range(num_decoder_layers):
      decoder_layers.append(
        TransformerDecoderBlock(
          d_model=d_model,
          nhead=nhead,
          dim_feedforward=dim_feedforward,
          dropout=dropout,
          activation=activation,
        )
      )
    self.transformer_decoder = nn.ModuleList(decoder_layers)

    # Output projection: d_model back to feature_dim
    self.output_projection = nn.Linear(d_model, feature_dim)

    # Initialize weights
    init_autoencoder_weights(self)

  def encode_with_memory(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode trajectory to latent vector and return encoder output for decoder.

    Args:
        x: Input trajectory of shape (batch_size, horizon, feature_dim)

    Returns:
        Tuple of (latent_vector, encoder_output)
        - latent_vector: (batch_size, latent_dim)
        - encoder_output: (horizon, batch_size, d_model) for use as decoder memory
    """
    # x: (batch_size, horizon, feature_dim)

    # Treat as 2D image: add channel dimension
    x_2d = x.unsqueeze(1)  # (batch_size, 1, horizon, feature_dim)

    # Apply 2D CNN
    cnn_features = self.cnn_encoder(
      x_2d
    )  # (batch_size, cnn_output_channels, horizon, feature_dim)

    # Concatenate CNN features with original image
    # CNN features: (batch, cnn_output_channels, horizon, feature_dim)
    # Original: (batch, 1, horizon, feature_dim) - need to expand
    x_2d_expanded = x_2d  # (batch, 1, horizon, feature_dim)

    # Concatenate along channel dimension: (batch, 1 + cnn_output_channels, horizon, feature_dim)
    if self.use_residual_connection:
      concatenated = torch.cat(
        [x_2d_expanded, cnn_features], dim=1
      )  # (batch, 1 + cnn_output_channels, horizon, feature_dim)
    else:
      concatenated = cnn_features  # (batch, cnn_output_channels, horizon, feature_dim)

    # Project to d_model: treat each spatial location independently
    # Reshape: (batch, channels, horizon, feature_dim) -> (batch, horizon, feature_dim, channels)
    concatenated_flat = concatenated.permute(
      0, 2, 3, 1
    )  # (batch, horizon, feature_dim, channels)

    # Project: (batch, horizon, feature_dim, channels) -> (batch, horizon, feature_dim, d_model)
    projected = self.cnn_projection(
      concatenated_flat
    )  # (batch, horizon, feature_dim, d_model)

    # Reshape to sequence: use horizon as sequence length, average over feature_dim
    # This gives us (batch, horizon, d_model) - one token per timestep
    combined = projected.mean(dim=2)  # (batch, horizon, d_model)

    # Prepare for transformer: (batch, horizon, d_model) -> (horizon, batch, d_model)
    combined = combined.transpose(0, 1)  # (horizon, batch, d_model)
    combined = combined * math.sqrt(self.d_model)  # Scale
    combined = self.pos_encoder(combined)  # (horizon, batch, d_model)

    # Pass through transformer encoder layers
    encoder_output = combined
    for encoder_layer in self.transformer_encoder:
      encoder_output = encoder_layer(encoder_output)

    # Global pooling: (horizon, batch, d_model) -> (batch, d_model, horizon) -> (batch, d_model)
    pooled = encoder_output.transpose(0, 1).transpose(1, 2)  # (batch, d_model, horizon)
    pooled = self.bottleneck_pool(pooled).squeeze(-1)  # (batch, d_model)

    # Compress to latent
    latent = self.bottleneck_mlp(pooled)  # (batch, latent_dim)

    return latent, encoder_output

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector.

    Args:
        x: Input trajectory of shape (batch_size, horizon, feature_dim)

    Returns:
        Latent vector of shape (batch_size, latent_dim)
    """
    latent, _ = self.encode_with_memory(x)
    return latent

  def decode(
    self, latent: torch.Tensor, encoder_output: torch.Tensor | None = None
  ) -> torch.Tensor:
    """Decode latent vector to trajectory.

    Args:
        latent: Latent vector of shape (batch_size, latent_dim)
        encoder_output: Encoder output of shape (horizon, batch_size, d_model) to use as memory.
                       If None, uses the expanded sequence as memory (fallback).

    Returns:
        Reconstructed trajectory of shape (batch_size, horizon, feature_dim)
    """
    batch_size = latent.shape[0]

    # Expand latent to sequence
    x = self.decoder_expand(latent)  # (batch_size, d_model * horizon)
    x = x.view(batch_size, self.horizon, self.d_model)  # (batch_size, horizon, d_model)
    x = x.transpose(0, 1)  # (horizon, batch_size, d_model)

    # Add positional encoding
    x = self.pos_decoder(x)  # (horizon, batch_size, d_model)

    # Use encoder output as memory if provided, otherwise use expanded sequence as fallback
    if encoder_output is not None:
      memory = encoder_output  # (horizon, batch_size, d_model) - from encoder
    else:
      memory = x  # Fallback: use expanded sequence as memory

    # Pass through transformer decoder layers
    for decoder_layer in self.transformer_decoder:
      x = decoder_layer(x, memory)

    # Project back to feature_dim
    x = self.output_projection(x)  # (horizon, batch_size, feature_dim)

    # Transpose back to (batch_size, horizon, feature_dim)
    x = x.transpose(0, 1)  # (batch_size, horizon, feature_dim)

    return x

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass: encode and decode.

    Args:
        x: Input trajectory of shape (batch_size, horizon, feature_dim)

    Returns:
        Tuple of (reconstructed_trajectory, latent_vector)
    """
    # Encode and get both latent and encoder output for decoder memory
    latent, encoder_output = self.encode_with_memory(x)
    reconstructed = self.decode(latent, encoder_output)
    return reconstructed, latent
