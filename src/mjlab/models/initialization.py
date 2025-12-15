"""Weight initialization utilities for neural networks."""

from __future__ import annotations

import torch.nn as nn


def init_weights(module: nn.Module, init_type: str = "kaiming") -> None:
  """Initialize weights for a module based on its type.

  Args:
      module: PyTorch module to initialize
      init_type: Type of initialization ('kaiming', 'xavier', or 'default')
  """
  if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)):
    if init_type == "kaiming":
      # Kaiming (He) initialization for ReLU activations
      nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    elif init_type == "xavier":
      # Xavier (Glorot) initialization
      nn.init.xavier_normal_(module.weight)
    # Default PyTorch initialization is already Kaiming for conv layers

    # Initialize biases to zero
    if module.bias is not None:
      nn.init.constant_(module.bias, 0.0)

  elif isinstance(module, nn.Linear):
    if init_type == "kaiming":
      # Kaiming initialization for ReLU activations
      nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")
    elif init_type == "xavier":
      # Xavier initialization
      nn.init.xavier_normal_(module.weight)
    # Default PyTorch initialization is already good for linear layers

    # Initialize biases to zero
    if module.bias is not None:
      nn.init.constant_(module.bias, 0.0)

  elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
    # BatchNorm: weight=1, bias=0 (default is already correct)
    nn.init.constant_(module.weight, 1.0)
    nn.init.constant_(module.bias, 0.0)


def init_autoencoder_weights(
  model: nn.Module,
  final_layer_scale: float = 0.01,
) -> None:
  """Initialize weights for an autoencoder model with best practices.

  Uses Kaiming initialization for most layers (good for ReLU activations),
  and smaller initialization for final output layers to start with smaller outputs.

  Args:
      model: Autoencoder model to initialize
      final_layer_scale: Scale factor for final output layer weights (default: 0.01)
  """
  for name, module in model.named_modules():
    if isinstance(
      module, (nn.Conv1d, nn.Conv2d, nn.ConvTranspose1d, nn.ConvTranspose2d)
    ):
      # Check if this is a final output layer
      if "final" in name.lower() or "output" in name.lower():
        # Small initialization for final layer
        nn.init.normal_(module.weight, mean=0.0, std=final_layer_scale)
      else:
        # Kaiming initialization for intermediate conv layers
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

      if module.bias is not None:
        nn.init.constant_(module.bias, 0.0)

    elif isinstance(module, nn.Linear):
      # Check if this is a final output layer
      if "final" in name.lower() or "output" in name.lower():
        # Small initialization for final layer
        nn.init.normal_(module.weight, mean=0.0, std=final_layer_scale)
      else:
        # Kaiming initialization for linear layers (they're followed by ReLU)
        nn.init.kaiming_normal_(module.weight, mode="fan_in", nonlinearity="relu")

      if module.bias is not None:
        nn.init.constant_(module.bias, 0.0)

    elif isinstance(module, (nn.BatchNorm1d, nn.BatchNorm2d)):
      # BatchNorm: weight=1, bias=0
      nn.init.constant_(module.weight, 1.0)
      nn.init.constant_(module.bias, 0.0)
