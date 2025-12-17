"""Base class for trajectory autoencoders."""

from __future__ import annotations

from abc import ABC, abstractmethod

import torch
import torch.nn as nn

from mjlab.models.normalization import TrajectoryNormalizer


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
    normalizer: TrajectoryNormalizer | None = None,
  ):
    """Initialize the base autoencoder.

    Args:
        horizon: Number of timesteps in the trajectory (H)
        input_dim: Dimension of features per timestep (default: 72)
        latent_dim: Dimension of the latent vector
        normalizer: Optional normalizer to apply to inputs/outputs. If provided,
                   inputs are automatically normalized and outputs are denormalized.
    """
    super().__init__()
    self.horizon = horizon
    self.input_dim = input_dim
    self.latent_dim = latent_dim

    # Register normalizer as a submodule so it's part of state_dict
    if normalizer is not None:
      self.normalizer = normalizer
    else:
      # Register as None so it doesn't appear in state_dict
      self.normalizer = None

  def encode(self, x: torch.Tensor) -> torch.Tensor:
    """Encode trajectory to latent vector.

    If a normalizer is attached to the model, inputs are automatically normalized.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim)

    Returns:
        Latent vector of shape (batch_size, latent_dim)
    """
    # Normalize input if normalizer is present
    if self.normalizer is not None:
      x = self.normalizer(x)

    # Call the implementation method
    return self._encode_impl(x)

  @abstractmethod
  def _encode_impl(self, x: torch.Tensor) -> torch.Tensor:
    """Internal encode implementation (subclasses should implement this).

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim), already normalized if normalizer is present

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

    If a normalizer is attached to the model, inputs are automatically normalized.
    Outputs are returned in normalized space (for loss computation during training).

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim)

    Returns:
        Tuple of (reconstructed_trajectory_normalized, latent_vector)
        Reconstructed trajectory is in normalized space if normalizer is present.
    """
    # encode() handles normalization internally
    latent = self.encode(x)
    reconstructed_normalized = self.decode(latent)

    return reconstructed_normalized, latent

  def forward_denormalized(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Forward pass with denormalized output.

    Useful for inference when you want output in original scale.

    Args:
        x: Input trajectory of shape (batch_size, horizon, input_dim)

    Returns:
        Tuple of (reconstructed_trajectory_denormalized, latent_vector)
        Reconstructed trajectory is in original (non-normalized) space.
    """
    reconstructed_normalized, latent = self.forward(x)

    # Denormalize output if normalizer is present
    if self.normalizer is not None:
      reconstructed = self.normalizer.denormalize(reconstructed_normalized)
    else:
      reconstructed = reconstructed_normalized

    return reconstructed, latent
