#!/usr/bin/env python3
"""Run multiple Hydra experiments sequentially.

This script allows you to run multiple experiment configurations one after another.
Useful for hyperparameter sweeps or architecture comparisons.

Usage:
    python scripts/run_multiple_experiments.py --configs unet tcn 2dcnn
    python scripts/run_multiple_experiments.py --configs experiments/unet_large_latent experiments/tcn_small_latent
    python scripts/run_multiple_experiments.py --config-file experiments_list.txt
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def run_experiment(config_name: str, base_config: str = "config") -> bool:
  """Run a single experiment with the given config name.

  Args:
      config_name: Name of the config file (without .yaml extension)
                 Can be "unet", "tcn", or "experiments/unet_large_latent"
      base_config: Base config to use (default: "config", ignored if config_name is full path)

  Returns:
      True if experiment completed successfully, False otherwise
  """
  print(f"\n{'=' * 80}")
  print(f"Running experiment: {config_name}")
  print(f"{'=' * 80}\n")

  # Build the command
  # If config_name contains a slash, it's a subdirectory config
  # For Hydra, we can use config groups: experiments=unet_large_latent
  if "/" in config_name:
    parts = config_name.split("/")
    if len(parts) == 2:
      # e.g., "experiments/unet_large_latent" -> experiments=unet_large_latent
      config_group = parts[0]  # "experiments"
      config_name_only = parts[1]  # "unet_large_latent"
      cmd = [
        sys.executable,
        "-m",
        "mjlab.scripts.train_trajectory_autoencoder_hydra",
        f"{config_group}={config_name_only}",
      ]
    else:
      # Fallback: use as direct config name (shouldn't happen with current structure)
      cmd = [
        sys.executable,
        "-m",
        "mjlab.scripts.train_trajectory_autoencoder_hydra",
        "--config-name",
        config_name,
      ]
  else:
    # Simple config name (e.g., "unet", "tcn", "2dcnn")
    cmd = [
      sys.executable,
      "-m",
      "mjlab.scripts.train_trajectory_autoencoder_hydra",
      "--config-name",
      config_name,
    ]

  print(f"Command: {' '.join(cmd)}\n")

  try:
    subprocess.run(cmd, check=True, cwd=Path(__file__).parent.parent)
    print(f"\n✓ Experiment '{config_name}' completed successfully!")
    return True
  except subprocess.CalledProcessError as e:
    print(f"\n✗ Experiment '{config_name}' failed with exit code {e.returncode}")
    return False
  except KeyboardInterrupt:
    print(f"\n⚠ Experiment '{config_name}' interrupted by user")
    return False


def main():
  parser = argparse.ArgumentParser(
    description="Run multiple Hydra experiments sequentially",
    formatter_class=argparse.RawDescriptionHelpFormatter,
    epilog="""
Examples:
  # Run basic architecture comparisons
  python scripts/run_multiple_experiments.py --configs unet tcn 2dcnn
  
  # Run custom experiment configs
  python scripts/run_multiple_experiments.py --configs experiments/unet_large_latent experiments/tcn_small_latent
  
  # Run from a file list
  python scripts/run_multiple_experiments.py --config-file experiments_list.txt
  
  # Run with stop-on-error
  python scripts/run_multiple_experiments.py --configs unet tcn --stop-on-error
        """,
  )
  parser.add_argument(
    "--configs",
    nargs="+",
    help="List of config names to run (without .yaml extension)",
  )
  parser.add_argument(
    "--config-file",
    type=str,
    help="Path to a text file containing config names (one per line)",
  )
  parser.add_argument(
    "--base-config",
    type=str,
    default="config",
    help="Base config to use (default: config)",
  )
  parser.add_argument(
    "--stop-on-error",
    action="store_true",
    help="Stop running experiments if one fails",
  )
  parser.add_argument(
    "--dry-run",
    action="store_true",
    help="Print commands without executing them",
  )

  args = parser.parse_args()

  # Collect config names
  configs = []
  if args.configs:
    configs.extend(args.configs)
  if args.config_file:
    config_file = Path(args.config_file)
    if not config_file.exists():
      print(f"Error: Config file not found: {config_file}")
      sys.exit(1)
    with open(config_file) as f:
      configs.extend(
        [line.strip() for line in f if line.strip() and not line.startswith("#")]
      )

  if not configs:
    print("Error: No configs specified. Use --configs or --config-file")
    sys.exit(1)

  print(f"Will run {len(configs)} experiments:")
  for i, config in enumerate(configs, 1):
    print(f"  {i}. {config}")

  if args.dry_run:
    print("\n[DRY RUN] Would execute:")
    for config in configs:
      print(
        f"  python -m mjlab.scripts.train_trajectory_autoencoder_hydra --config-name {config}"
      )
    return

  # Run experiments
  results = []
  for i, config in enumerate(configs, 1):
    print(f"\n[{i}/{len(configs)}] Starting experiment: {config}")
    success = run_experiment(config, args.base_config)
    results.append((config, success))

    if not success and args.stop_on_error:
      print(f"\nStopping due to error in experiment: {config}")
      break

  # Summary
  print(f"\n{'=' * 80}")
  print("Summary:")
  print(f"{'=' * 80}")
  successful = sum(1 for _, success in results if success)
  failed = len(results) - successful
  for config, success in results:
    status = "✓" if success else "✗"
    print(f"  {status} {config}")
  print(f"\nTotal: {len(results)} experiments")
  print(f"  Successful: {successful}")
  print(f"  Failed: {failed}")


if __name__ == "__main__":
  main()
