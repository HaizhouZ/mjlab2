# Trajectory Autoencoder Training Configurations

This directory contains Hydra configuration files for training trajectory autoencoders.

## Quick Start

### Single Experiment

Run a single experiment with a specific config:

```bash
# From project root
python -m mjlab.scripts.train_trajectory_autoencoder_hydra --config-name unet

# Or with custom overrides
python -m mjlab.scripts.train_trajectory_autoencoder_hydra --config-name config \
    architecture=tcn latent_dim=256 epochs=500
```

### Multiple Experiments Sequentially

Use the helper script to run multiple configs one after another:

```bash
# Run basic architecture comparisons
python scripts/run_multiple_experiments.py --configs unet tcn 2dcnn

# Run custom experiment configs
python scripts/run_multiple_experiments.py --configs experiments/unet_large_latent experiments/tcn_small_latent

# Run from a file list
python scripts/run_multiple_experiments.py --config-file experiments_list.txt

# Stop on first error
python scripts/run_multiple_experiments.py --configs unet tcn --stop-on-error
```

## Configuration Files

### Base Config (`config.yaml`)
The base configuration file with all default settings. Other configs override specific values.

### Architecture Configs
- `unet.yaml` - U-Net architecture
- `tcn.yaml` - TCN architecture  
- `2dcnn.yaml` - 2D CNN architecture

### Experiment Configs (`experiments/`)
Pre-configured experiments for common hyperparameter sweeps:
- `unet_large_latent.yaml` - U-Net with larger latent dimension (256)
- `tcn_small_latent.yaml` - TCN with smaller latent dimension (64)
- `horizon64.yaml` - Longer horizon (64 timesteps)
- `onecycle_scheduler.yaml` - OneCycleLR scheduler

## Creating Custom Configs

You can create custom config files by:

1. **Creating a new YAML file** in `conf/trajectory_autoencoder/` or `conf/trajectory_autoencoder/experiments/`
2. **Using `@override config`** to inherit from base config
3. **Overriding specific parameters**

Example (`conf/trajectory_autoencoder/experiments/my_experiment.yaml`):

```yaml
# @package _global_
# @override config

architecture: "unet"
latent_dim: 512
horizon: 64
batch_size: 64
lr: 0.0005
scheduler: "onecycle"
max_lr: 0.005
output_dir: "checkpoints/trajectory_autoencoder/my_experiment"
wandb_run_name: "my_experiment_latent512"
```

## Configuration Parameters

### Data
- `motion_dir`: Base directory containing trajectory subdirectories
- `traj_patterns`: List of regex patterns to match trajectory names
- `horizon`: Number of timesteps in trajectory window
- `stride`: Stride for sliding window extraction
- `val_split`: Fraction of data for validation

### Model
- `architecture`: Architecture type (`unet`, `tcn`, `2dcnn`)
- `latent_dim`: Dimension of latent vector
- `input_dim`: Input dimension (usually computed from data)

### Training
- `batch_size`: Batch size
- `epochs`: Number of training epochs
- `lr`: Learning rate
- `device`: Device to train on (`cuda` or `cpu`)

### Learning Rate Scheduler
- `scheduler`: Scheduler type (`none`, `onecycle`, `cosine_annealing`, `cosine_restarts`, `reduce_on_plateau`, `warmup_cosine`)
- `warmup_epochs`: Number of warmup epochs (for `warmup_cosine`)
- `max_lr`: Maximum learning rate (for `onecycle`)
- `min_lr`: Minimum learning rate (for cosine schedulers)
- `t_0`: Initial period (for `cosine_restarts`)
- `t_mult`: Period multiplier (for `cosine_restarts`)

### Normalization
- `normalize`: Whether to normalize inputs
- `normalizer_file`: Path to save/load normalizer (null = auto)

### Output
- `output_dir`: Directory to save checkpoints

### Wandb
- `no_wandb`: Disable wandb logging
- `wandb_project`: Wandb project name
- `wandb_run_name`: Wandb run name (null = auto)

## Hydra Outputs

Hydra automatically creates output directories for each run:
- Default: `outputs/YYYY-MM-DD/HH-MM-SS/`
- Contains: config, logs, and any files saved during training

To change this, use `--hydra.run.dir`:

```bash
python -m mjlab.scripts.train_trajectory_autoencoder_hydra \
    --config-name unet \
    hydra.run.dir=./my_custom_output
```

## Tips

1. **Use wandb for tracking**: Enable wandb to compare experiments easily
2. **Start small**: Test with fewer epochs first (`epochs=10`) to verify configs
3. **Use experiments/ directory**: Organize custom configs in the `experiments/` subdirectory
4. **Sequential runs**: Use `run_multiple_experiments.py` for hyperparameter sweeps
5. **Checkpoints**: Models are saved to `output_dir` specified in config

