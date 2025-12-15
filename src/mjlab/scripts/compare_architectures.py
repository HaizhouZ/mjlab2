"""Compare different autoencoder architectures."""

from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

from mjlab.models import (
  TrajectoryAutoencoder2DCNN,
  TrajectoryAutoencoderTCN,
  TrajectoryAutoencoderUNet,
)


def compare_architectures():
  """Compare different architectures."""
  batch_size = 8
  horizon = 32
  feature_dim = 72
  latent_dim = 128

  architectures = [
    ("U-Net", TrajectoryAutoencoderUNet),
    ("TCN", TrajectoryAutoencoderTCN),
    ("2D CNN", TrajectoryAutoencoder2DCNN),
  ]

  print("=" * 80)
  print("Architecture Comparison")
  print("=" * 80)
  print(
    f"Input: (batch_size={batch_size}, horizon={horizon}, feature_dim={feature_dim})"
  )
  print(f"Latent dimension: {latent_dim}\n")

  results = []

  for name, model_class in architectures:
    print(f"{'=' * 80}")
    print(f"{name}")
    print(f"{'=' * 80}")

    # Create model
    model = model_class(
      horizon=horizon,
      feature_dim=feature_dim,
      latent_dim=latent_dim,
    )

    # Count parameters
    num_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    # Test forward pass
    x = torch.randn(batch_size, horizon, feature_dim)
    with torch.no_grad():
      reconstructed, latent = model(x)

    # Compute reconstruction error (untrained)
    mse = nn.MSELoss()(reconstructed, x).item()

    # Memory usage estimate (rough)
    model_size_mb = num_params * 4 / (1024 * 1024)  # Assuming float32

    print(f"Parameters: {num_params:,} ({trainable_params:,} trainable)")
    print(f"Model size: ~{model_size_mb:.2f} MB")
    print(f"Latent shape: {latent.shape}")
    print(f"Reconstruction MSE (untrained): {mse:.6f}")

    # Test inference time
    import time

    model.eval()
    with torch.no_grad():
      # Warmup
      for _ in range(10):
        _ = model(x)

      # Time it
      start = time.time()
      for _ in range(100):
        _ = model(x)
      elapsed = time.time() - start
      avg_time_ms = (elapsed / 100) * 1000

    print(f"Avg inference time: {avg_time_ms:.3f} ms/batch")

    results.append(
      {
        "name": name,
        "params": num_params,
        "size_mb": model_size_mb,
        "mse": mse,
        "inference_ms": avg_time_ms,
      }
    )

    print()

  # Summary table
  print("=" * 80)
  print("Summary")
  print("=" * 80)
  print(
    f"{'Architecture':<15} {'Params':<15} {'Size (MB)':<12} {'MSE':<12} {'Inference (ms)':<15}"
  )
  print("-" * 80)
  for r in results:
    print(
      f"{r['name']:<15} {r['params']:>14,} {r['size_mb']:>11.2f} {r['mse']:>11.6f} {r['inference_ms']:>14.3f}"
    )

  print("\nRecommendations:")
  print("- U-Net: Best balance of speed, size, and quality")
  print("- TCN: Best for long-range temporal dependencies")
  print("- 2D CNN: Best when spatial feature relationships matter")


if __name__ == "__main__":
  compare_architectures()
