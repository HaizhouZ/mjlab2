"""Normalization utilities for trajectory data."""

from __future__ import annotations

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

  def state_dict(self, *args, **kwargs) -> dict[str, torch.Tensor]:
    """Get state dict for saving.

    Compatible with PyTorch's standard state_dict() signature.
    """
    # Call parent to get standard buffers (mean, std)
    state = super().state_dict(*args, **kwargs)

    # Add epsilon to the state dict
    # Handle prefix from kwargs
    prefix = kwargs.get("prefix", "")
    if prefix:
      state[f"{prefix}epsilon"] = torch.tensor(self.epsilon)
    else:
      state["epsilon"] = torch.tensor(self.epsilon)

    return state

  def load_state_dict(
    self,
    state_dict,
    strict: bool = True,
    assign: bool = False,
  ):
    """Load state dict.

    Compatible with PyTorch's standard load_state_dict() signature.
    Returns _IncompatibleKeys object with missing_keys and unexpected_keys.
    """
    # Make a copy to avoid modifying the original
    state_dict = dict(state_dict)

    # Extract epsilon if present (handle with or without prefix)
    epsilon_value = None
    for key in list(state_dict.keys()):
      if key.endswith(".epsilon") or key == "epsilon":
        epsilon_value = state_dict.pop(key)
        break

    # Load standard buffers (mean, std) via parent
    incompatible_keys = super().load_state_dict(
      state_dict, strict=strict, assign=assign
    )

    # Restore epsilon if it was in the state dict
    if epsilon_value is not None:
      if isinstance(epsilon_value, torch.Tensor):
        self.epsilon = epsilon_value.item()
      else:
        self.epsilon = float(epsilon_value)

    return incompatible_keys
