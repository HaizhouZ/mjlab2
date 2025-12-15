"""Transformer-based autoencoder for trajectory data.

Uses Transformer encoder-decoder architecture with self-attention
to capture long-range temporal dependencies in trajectory sequences.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from mjlab.models.initialization import init_autoencoder_weights
from mjlab.models.trajectory_autoencoder_base import TrajectoryAutoencoderBase


class PositionalEncoding(nn.Module):
  """Positional encoding for transformer."""

  def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.1):
    super().__init__()
    self.dropout = nn.Dropout(p=dropout)

    position = torch.arange(max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model))
    pe = torch.zeros(max_len, 1, d_model)
    pe[:, 0, 0::2] = torch.sin(position * div_term)
    pe[:, 0, 1::2] = torch.cos(position * div_term)
    self.register_buffer("pe", pe)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """
    Args:
        x: Tensor, shape [seq_len, batch_size, embedding_dim]
    """
    x = x + self.pe[: x.size(0)]
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


class TrajectoryAutoencoderTransformer(TrajectoryAutoencoderBase):
  """Transformer-based autoencoder for compressing trajectory data.

  Uses Transformer encoder-decoder architecture:
  - Encoder: Stack of transformer encoder blocks with positional encoding
  - Bottleneck: Compress sequence to latent vector
  - Decoder: Stack of transformer decoder blocks to reconstruct sequence

  Architecture:
  - Input embedding: Linear projection of input_dim to d_model
  - Encoder: N transformer encoder blocks
  - Bottleneck: Global pooling + MLP to latent_dim
  - Decoder: Expand latent to sequence + N transformer decoder blocks
  - Output: Linear projection back to input_dim
  """

  def __init__(
    self,
    horizon: int,
    input_dim: int = 72,
    latent_dim: int = 128,
    d_model: int = 256,
    nhead: int = 8,
    num_encoder_layers: int = 4,
    num_decoder_layers: int = 4,
    dim_feedforward: int = 1024,
    dropout: float = 0.1,
    activation: str = "relu",
  ):
    """Initialize the transformer autoencoder.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        input_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the latent vector
        d_model: Dimension of transformer model (default: 256)
        nhead: Number of attention heads (default: 8)
        num_encoder_layers: Number of encoder layers (default: 4)
        num_decoder_layers: Number of decoder layers (default: 4)
        dim_feedforward: Dimension of feedforward network (default: 1024)
        dropout: Dropout rate (default: 0.1)
        activation: Activation function ('relu' or 'gelu', default: 'relu')
    """
    super().__init__(horizon, input_dim, latent_dim)

    self.d_model = d_model

    # Input embedding: project input_dim to d_model
    self.input_embedding = nn.Linear(input_dim, d_model)

    # Positional encoding
    self.pos_encoder = PositionalEncoding(d_model, max_len=horizon * 2, dropout=dropout)
    self.pos_decoder = PositionalEncoding(d_model, max_len=horizon * 2, dropout=dropout)

    # Encoder: stack of transformer encoder blocks
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
    self.encoder = nn.ModuleList(encoder_layers)

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

    # Decoder: stack of transformer decoder blocks
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
    self.decoder = nn.ModuleList(decoder_layers)

    # Output projection: d_model back to input_dim
    self.output_projection = nn.Linear(d_model, input_dim)

    # Initialize weights
    init_autoencoder_weights(self)

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim)

    Returns:
        Latent vector of shape (batch_size, latent_dim)
    """
    # x: (batch_size, horizon, input_dim)

    # Embed and add positional encoding
    # Transformer expects (seq_len, batch_size, d_model)
    x = x.transpose(0, 1)  # (horizon, batch_size, input_dim)
    x = self.input_embedding(x) * math.sqrt(
      self.d_model
    )  # (horizon, batch_size, d_model)
    x = self.pos_encoder(x)  # (horizon, batch_size, d_model)

    # Pass through encoder layers
    for encoder_layer in self.encoder:
      x = encoder_layer(x)

    # Global pooling: (horizon, batch_size, d_model) -> (batch_size, d_model, horizon) -> (batch_size, d_model)
    x = x.transpose(0, 1).transpose(1, 2)  # (batch_size, d_model, horizon)
    x = self.bottleneck_pool(x).squeeze(-1)  # (batch_size, d_model)

    # Compress to latent
    latent = self.bottleneck_mlp(x)  # (batch_size, latent_dim)

    return latent

  def decode(self, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent vector to trajectory.

    Args:
        latent: Latent vector of shape (batch_size, latent_dim)

    Returns:
        Reconstructed trajectory of shape (batch_size, horizon, input_dim)
    """
    batch_size = latent.shape[0]

    # Expand latent to sequence
    x = self.decoder_expand(latent)  # (batch_size, d_model * horizon)
    x = x.view(batch_size, self.horizon, self.d_model)  # (batch_size, horizon, d_model)
    x = x.transpose(0, 1)  # (horizon, batch_size, d_model)

    # Add positional encoding
    x = self.pos_decoder(x)  # (horizon, batch_size, d_model)

    # Use encoder output as memory (we'll use a learned query or the expanded sequence)
    # For simplicity, we'll use the expanded sequence as both query and memory
    # In a full implementation, we might want to use the encoder output as memory
    memory = x  # Use expanded sequence as memory

    # Pass through decoder layers
    for decoder_layer in self.decoder:
      x = decoder_layer(x, memory)

    # Project back to input_dim
    x = self.output_projection(x)  # (horizon, batch_size, input_dim)

    # Transpose back to (batch_size, horizon, input_dim)
    x = x.transpose(0, 1)  # (batch_size, horizon, input_dim)

    return x
