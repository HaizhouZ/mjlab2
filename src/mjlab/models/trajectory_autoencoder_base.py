"""Base class for trajectory autoencoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn


class TrajectoryAutoencoderBase(nn.Module, ABC):
  """Base class for trajectory autoencoders.

  All autoencoder implementations should inherit from this class
  and implement encode() and decode() methods.
  """

  def __init__(
    self,
    horizon: int,
    input_dim: int = 72,
    latent_dim: int = 128,
  ):
    """Initialize the base autoencoder.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        input_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the latent vector
    """
    super().__init__()
    self.horizon = horizon
    self.input_dim = input_dim
    self.latent_dim = latent_dim

  @abstractmethod
  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim)

    Returns:
        Latent vector of shape (batch_size, latent_dim)
    """
    pass

  @abstractmethod
  def decode(self, latent: torch.Tensor) -> torch.Tensor:
    """Decode latent vector to trajectory.

    Args:
        latent: Latent vector of shape (batch_size, latent_dim)

    Returns:
        Reconstructed trajectory of shape (batch_size, horizon, input_dim)
    """
    pass

  def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass: encode and decode.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim)

    Returns:
        Tuple of (reconstructed_trajectory, latent_vector)
    """
    latent = self.encode(x)
    reconstructed = self.decode(latent)
    return reconstructed, latent
