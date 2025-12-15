# Trajectory Autoencoder

Multiple autoencoder architectures for compressing robot trajectory data to a latent representation and reconstructing it back.

## Overview

The trajectory autoencoders compress temporal robot trajectory data (joint positions, velocities, anchor pose, object pose) into a compact latent vector and can reconstruct the full trajectory from the latent.

## Available Architectures

### 1. U-Net (`TrajectoryAutoencoderUNet`)
- **Architecture**: 1D convolutional layers with stride=2 downsampling
- **Best for**: General purpose, balanced performance
- **Parameters**: ~1.1M (with default config)
- **Pros**: Simple, efficient, good reconstruction quality
- **Cons**: May struggle with very long sequences

### 2. TCN (`TrajectoryAutoencoderTCN`)
- **Architecture**: Temporal Convolutional Network with dilated convolutions
- **Best for**: Long-range temporal dependencies, sequences with varying temporal scales
- **Parameters**: ~5M (with default config)
- **Pros**: Captures long-range dependencies, maintains temporal resolution
- **Cons**: More parameters, slower training

### 3. 2D CNN (`TrajectoryAutoencoder2DCNN`)
- **Architecture**: Treats trajectory as 2D image (time × features)
- **Best for**: When spatial relationships between features matter
- **Parameters**: ~2.8M (with default config)
- **Pros**: Can capture 2D patterns in the trajectory matrix
- **Cons**: More memory intensive

### Input Features (per timestep, total 72 dimensions)
- `joint_pos`: 29 dimensions (joint positions)
- `joint_vel`: 29 dimensions (joint velocities)
- `anchor_pos`: 3 dimensions (robot anchor body position)
- `anchor_quat`: 4 dimensions (robot anchor body quaternion)
- `object_pos`: 3 dimensions (object position)
- `object_quat`: 4 dimensions (object quaternion)

### Architecture
- **Encoder**: 1D convolutional layers with stride=2 downsampling
- **Bottleneck**: Fully connected layers compressing to latent dimension
- **Decoder**: Transpose convolutional layers with upsampling back to original horizon

## Usage

### Basic Usage

```python
import torch
from mjlab.models import (
    TrajectoryAutoencoderUNet,      # U-Net
    TrajectoryAutoencoderTCN,  # TCN
    TrajectoryAutoencoder2DCNN, # 2D CNN
)

# Create U-Net model
model_unet = TrajectoryAutoencoderUNet(
    horizon=32,        # Number of timesteps in trajectory
    input_dim=72,    # Features per timestep
    latent_dim=128,    # Latent vector dimension
)

# Create TCN model
model_tcn = TrajectoryAutoencoderTCN(
    horizon=32,
    input_dim=72,
    latent_dim=128,
)

# Create 2D CNN model
model_2dcnn = TrajectoryAutoencoder2DCNN(
    horizon=32,
    input_dim=72,
    latent_dim=128,
)

# Use any model the same way
trajectory = torch.randn(4, 32, 72)  # (batch_size, horizon, input_dim)
latent = model_unet.encode(trajectory)    # (batch_size, latent_dim)
reconstructed = model_unet.decode(latent)  # (batch_size, horizon, input_dim)

# Or use forward pass
reconstructed, latent = model_unet(trajectory)
```

### Training

Use the provided training script with different architectures:

```bash
# Train U-Net (default)
python src/mjlab/scripts/train_trajectory_autoencoder.py \
    --motion-dir motions/output/difficult_sampling \
    --architecture unet \
    --horizon 32 \
    --latent-dim 128 \
    --batch-size 32 \
    --epochs 100 \
    --device cuda

# Train TCN
python src/mjlab/scripts/train_trajectory_autoencoder.py \
    --motion-dir motions/output/difficult_sampling \
    --architecture tcn \
    --horizon 32 \
    --latent-dim 128 \
    --batch-size 32 \
    --epochs 100 \
    --device cuda

# Train 2D CNN
python src/mjlab/scripts/train_trajectory_autoencoder.py \
    --motion-dir motions/output/difficult_sampling \
    --architecture 2dcnn \
    --horizon 32 \
    --latent-dim 128 \
    --batch-size 32 \
    --epochs 100 \
    --device cuda
```

### Loading Data

The `TrajectoryDataset` class uses `MultiMotionLoader` to load trajectory windows:

```python
from mjlab.models.trajectory_dataset import TrajectoryDataset
import torch

# Create dataset
dataset = TrajectoryDataset(
    motion_dir="motions/output/difficult_sampling",
    traj_name_patterns=[".*"],
    body_indexes=torch.arange(14),  # Indices for 14 tracked bodies
    anchor_body_name="torso_link",
    body_names=(
        "pelvis", "left_hip_roll_link", "left_knee_link", 
        # ... etc
    ),
    horizon=32,
    device="cpu",
)

# Get a trajectory window
trajectory = dataset[0]  # Shape: (horizon, input_dim)
```

## Model Parameters

### Common Parameters (all architectures)
- `horizon`: Number of timesteps in each trajectory window
- `input_dim`: Dimension of features per timestep (default: 72)
- `latent_dim`: Dimension of compressed latent vector (default: 128)

### U-Net Specific
- `encoder_channels`: List of channel sizes for encoder (default: [64, 128, 256])
- `decoder_channels`: List of channel sizes for decoder (default: [256, 128, 64])
- `use_batch_norm`: Whether to use batch normalization (default: True)

### TCN Specific
- `num_channels`: List of channel sizes for TCN blocks (default: [64, 128, 256])
- `kernel_size`: Kernel size for temporal convolutions (default: 3)
- `dropout`: Dropout rate (default: 0.2)

### 2D CNN Specific
- `encoder_channels`: List of channel sizes for encoder (default: [32, 64, 128])
- `decoder_channels`: List of channel sizes for decoder (default: [128, 64, 32])
- `kernel_size`: Kernel size for convolutions (int or tuple, default: 3)
- `use_batch_norm`: Whether to use batch normalization (default: True)

## Files

- `trajectory_autoencoder.py`: Main autoencoder model
- `trajectory_dataset.py`: Dataset loader using MultiMotionLoader
- `train_trajectory_autoencoder.py`: Training script
- `test_trajectory_autoencoder.py`: Test script

