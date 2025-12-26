"""Normalization utilities for trajectory data."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn


class TrajectoryNormalizer(nn.Module):
  """Normalizes trajectory data using mean and standard deviation.

  Computes statistics from training data and normalizes inputs to have
  zero mean and unit variance per feature dimension.
  """

  def __init__(
    self,
    mean: torch.Tensor | None = None,
    std: torch.Tensor | None = None,
    epsilon: float = 1e-8,
  ):
    """Initialize the normalizer.

    Args:
        mean: Mean tensor of shape (input_dim,) or (horizon, input_dim)
        std: Standard deviation tensor of shape (input_dim,) or (horizon, input_dim)
        epsilon: Small value to avoid division by zero
    """
    super().__init__()
    self.epsilon = epsilon

    if mean is not None and std is not None:
      self.register_buffer("mean", mean)
      self.register_buffer("std", std)
    else:
      self.register_buffer("mean", None)
      self.register_buffer("std", None)

  def fit(self, data: torch.Tensor) -> None:
    """Compute normalization statistics from data.

    Args:
        data: Input data of shape (num_samples, horizon, input_dim) or (num_samples, input_dim)
    """
    # Compute mean and std across all samples and time steps
    if data.ndim == 3:
      # (num_samples, horizon, input_dim) -> compute over samples and horizon
      mean = data.mean(dim=(0, 1))  # (input_dim,)
      std = data.std(dim=(0, 1))  # (input_dim,)
    elif data.ndim == 2:
      # (num_samples, input_dim) -> compute over samples
      mean = data.mean(dim=0)  # (input_dim,)
      std = data.std(dim=0)  # (input_dim,)
    else:
      raise ValueError(f"Expected 2D or 3D tensor, got {data.ndim}D")

    # Avoid division by zero
    std = torch.clamp(std, min=self.epsilon)

    # Move to same device as data
    mean = mean.to(data.device)
    std = std.to(data.device)

    self.register_buffer("mean", mean)
    self.register_buffer("std", std)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    """Normalize input data.

    Args:
        x: Input tensor of shape (..., horizon, input_dim) or (..., input_dim)

    Returns:
        Normalized tensor with same shape
    """
    if self.mean is None or self.std is None:
      raise RuntimeError("Normalizer not fitted. Call fit() first or provide mean/std.")

    # Expand mean and std to match input dimensions
    # x: (batch, horizon, input_dim) or (batch, input_dim)
    # mean/std: (input_dim,)
    # We need to expand to (1, 1, input_dim) or (1, input_dim)
    if x.ndim == 3:
      # (batch, horizon, input_dim)
      mean = self.mean.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
      std = self.std.unsqueeze(0).unsqueeze(0)  # (1, 1, input_dim)
    elif x.ndim == 2:
      # (batch, input_dim)
      mean = self.mean.unsqueeze(0)  # (1, input_dim)
      std = self.std.unsqueeze(0)  # (1, input_dim)
    else:
      raise ValueError(f"Expected 2D or 3D tensor, got {x.ndim}D")

    return (x - mean) / std

  def denormalize(self, x: torch.Tensor) -> torch.Tensor:
    """Denormalize data back to original scale.

    Args:
        x: Normalized tensor of shape (..., horizon, input_dim) or (..., input_dim)

    Returns:
        Denormalized tensor with same shape
    """
    if self.mean is None or self.std is None:
      raise RuntimeError("Normalizer not fitted. Call fit() first or provide mean/std.")

    # Expand mean and std to match input dimensions
    if x.ndim == 3:
      mean = self.mean.unsqueeze(0).unsqueeze(0)
      std = self.std.unsqueeze(0).unsqueeze(0)
    elif x.ndim == 2:
      mean = self.mean.unsqueeze(0)
      std = self.std.unsqueeze(0)
    else:
      raise ValueError(f"Expected 2D or 3D tensor, got {x.ndim}D")

    return x * std + mean

  def state_dict(
    self,
    destination: dict[str, Any] | None = None,
    prefix: str = "",
    keep_vars: bool = False,
  ) -> dict[str, Any]:
    """Get state dict for saving."""
    # Call parent to get buffers (mean, std) with proper handling of kwargs
    state_dict = super().state_dict(
      destination=destination, prefix=prefix, keep_vars=keep_vars
    )  # type: ignore[arg-type]

    # Add epsilon to state dict (it's not a buffer, so we add it manually)
    epsilon_key = prefix + "epsilon"
    epsilon_value = torch.tensor(self.epsilon)
    if not keep_vars:
      epsilon_value = epsilon_value.detach()

    if destination is not None:
      destination[epsilon_key] = epsilon_value
      return destination
    else:
      state_dict[epsilon_key] = epsilon_value
      return state_dict

  def load_state_dict(
    self,
    state_dict: Mapping[str, Any],
    strict: bool = True,
    assign: bool = False,
  ) -> Any:
    """Load state dict."""
    # Extract epsilon if present (it's not a buffer, so parent won't handle it)
    epsilon_value = None
    if "epsilon" in state_dict:
      epsilon_value = state_dict["epsilon"]
      # Create a new dict without epsilon for parent to process
      state_dict_without_epsilon = {
        k: v for k, v in state_dict.items() if k != "epsilon"
      }
    else:
      state_dict_without_epsilon = dict(state_dict)

    # Call parent to load buffers (mean, std)
    result = super().load_state_dict(
      state_dict_without_epsilon, strict=strict, assign=assign
    )

    # Load epsilon if present
    if epsilon_value is not None:
      if isinstance(epsilon_value, torch.Tensor):
        self.epsilon = epsilon_value.item()
      else:
        self.epsilon = float(epsilon_value)

    return result
