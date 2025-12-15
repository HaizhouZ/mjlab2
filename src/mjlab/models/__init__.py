"""Neural network models for mjlab."""

from mjlab.models.normalization import TrajectoryNormalizer
from mjlab.models.trajectory_autoencoder_2dcnn import TrajectoryAutoencoder2DCNN
from mjlab.models.trajectory_autoencoder_2dcnn_causal import (
  TrajectoryAutoencoder2DCNNCausal,
)
from mjlab.models.trajectory_autoencoder_base import TrajectoryAutoencoderBase
from mjlab.models.trajectory_autoencoder_tcn import TrajectoryAutoencoderTCN
from mjlab.models.trajectory_autoencoder_transformer import (
  TrajectoryAutoencoderTransformer,
)
from mjlab.models.trajectory_autoencoder_unet import TrajectoryAutoencoderUNet
from mjlab.models.trajectory_autoencoder_unet_residual import (
  TrajectoryAutoencoderUNetResidual,
)
from mjlab.models.trajectory_autoencoder_unet_simple import (
  TrajectoryAutoencoderUNetSimple,
)
from mjlab.models.trajectory_autoencoder_unet_simple_residual import (
  TrajectoryAutoencoderUNetSimpleResidual,
)
from mjlab.models.trajectory_autoencoder_unet_simple_temporal import (
  TrajectoryAutoencoderUNetSimpleTemporal,
)

__all__ = [
  "TrajectoryAutoencoderUNet",
  "TrajectoryAutoencoderUNetResidual",
  "TrajectoryAutoencoderUNetSimple",
  "TrajectoryAutoencoderUNetSimpleResidual",
  "TrajectoryAutoencoderUNetSimpleTemporal",
  "TrajectoryAutoencoder2DCNN",
  "TrajectoryAutoencoder2DCNNCausal",
  "TrajectoryAutoencoderBase",
  "TrajectoryAutoencoderTCN",
  "TrajectoryAutoencoderTransformer",
  "TrajectoryNormalizer",
]
