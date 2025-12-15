# Quick Start Guide: Hydra-based Training

## Installation

First, install Hydra and OmegaConf:

```bash
pip install hydra-core omegaconf
```

## Basic Usage

### Single Experiment

```bash
# Run with default config
python -m mjlab.scripts.train_trajectory_autoencoder_hydra

# Run with specific architecture
python -m mjlab.scripts.train_trajectory_autoencoder_hydra --config-name unet

# Override parameters on command line
python -m mjlab.scripts.train_trajectory_autoencoder_hydra --config-name unet \
    latent_dim=256 epochs=500 lr=0.0005
```

### Multiple Experiments Sequentially

```bash
# Run multiple configs one after another
python scripts/run_multiple_experiments.py --configs unet tcn 2dcnn

# Run experiments from a file
python scripts/run_multiple_experiments.py --config-file conf/trajectory_autoencoder/experiments_list.txt

# Stop on first error
python scripts/run_multiple_experiments.py --configs unet tcn --stop-on-error
```

## Example: Architecture Comparison

Create a file `my_experiments.txt`:

```
unet
tcn
2dcnn
```

Then run:

```bash
python scripts/run_multiple_experiments.py --config-file my_experiments.txt
```

This will train all three architectures sequentially, each with its own output directory and wandb run.

## Example: Hyperparameter Sweep

Create custom configs for different latent dimensions:

1. Create `conf/trajectory_autoencoder/experiments/latent64.yaml`:
```yaml
# @package _global_
# @override config
latent_dim: 64
output_dir: "checkpoints/trajectory_autoencoder/latent64"
wandb_run_name: "unet_latent64"
```

2. Create `conf/trajectory_autoencoder/experiments/latent256.yaml`:
```yaml
# @package _global_
# @override config
latent_dim: 256
output_dir: "checkpoints/trajectory_autoencoder/latent256"
wandb_run_name: "unet_latent256"
```

3. Run them:
```bash
python scripts/run_multiple_experiments.py --configs \
    experiments/latent64 \
    experiments/latent256
```

## Benefits of Hydra

1. **Reproducibility**: All configs are saved with each run
2. **Organization**: Easy to manage multiple experiments
3. **Flexibility**: Override any parameter from command line
4. **Sequential Runs**: Run multiple configs automatically
5. **Output Management**: Hydra organizes outputs by date/time

## Output Structure

Hydra creates organized output directories:

```
outputs/
  2024-01-15/
    10-30-45/  # Run 1
    11-15-20/  # Run 2
    12-00-10/  # Run 3
```

Each run contains:
- `.hydra/config.yaml` - Full config used
- `.hydra/hydra.yaml` - Hydra config
- `.hydra/overrides.yaml` - Command-line overrides
- Your model checkpoints (in `output_dir` from config)

## Tips

1. **Use wandb**: Compare experiments easily in wandb dashboard
2. **Start small**: Test with `epochs=10` first
3. **Use descriptive names**: Set `wandb_run_name` in configs
4. **Organize configs**: Put related experiments in `experiments/` subdirectory
5. **Check logs**: Each run's logs are in the Hydra output directory

