"""RSL-RL configuration."""

from dataclasses import dataclass, field
from typing import Literal, Tuple


@dataclass
class RslRlPpoActorCriticCfg:
  """Config for the PPO actor-critic networks."""

  init_noise_std: float = 1.0
  """The initial noise standard deviation of the policy."""
  noise_std_type: Literal["scalar", "log"] = "scalar"
  """The type of noise standard deviation for the policy. Default is scalar."""
  actor_obs_normalization: bool = False
  """Whether to normalize the observation for the actor network. Default is False."""
  critic_obs_normalization: bool = False
  """Whether to normalize the observation for the critic network. Default is False."""
  actor_hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the actor network."""
  critic_hidden_dims: Tuple[int, ...] = (128, 128, 128)
  """The hidden dimensions of the critic network."""
  activation: str = "elu"
  """The activation function to use in the actor and critic networks."""
  class_name: str = "ActorCritic"
  """Ignore, required by RSL-RL."""


@dataclass
class RslRlReppoActorCriticCfg(RslRlPpoActorCriticCfg):
  """Config for the official REPPO ActorQ network."""

  activation: str = "elu"
  """Activation function used by ActorQ."""
  num_critic_bins: int = 151
  """Number of bins in the categorical critic support."""
  vmin: float = -10.0
  """Minimum value of the categorical critic support."""
  vmax: float = 10.0
  """Maximum value of the categorical critic support."""
  state_dependent_std: bool = True
  """Whether ActorQ predicts state-dependent action std."""
  distribution_type: Literal["normal", "tanh"] = "tanh"
  """Policy action distribution type."""
  init_alpha_temp: float = 0.1
  """Initial entropy temperature."""
  init_alpha_kl: float = 0.1
  """Initial KL penalty multiplier."""
  action_lower_bound: float = -1.0
  """Lower bound used by the tanh action distribution."""
  action_upper_bound: float = 1.0
  """Upper bound used by the tanh action distribution."""
  class_name: str = "ActorQ"
  """Official REPPO actor-critic module class name."""


@dataclass
class RslRlPpoAlgorithmCfg:
  """Config for the PPO algorithm."""

  num_learning_epochs: int = 5
  """The number of learning epochs per update."""
  num_mini_batches: int = 4
  """The number of mini-batches per update.
  mini batch size = num_envs * num_steps / num_mini_batches
  """
  learning_rate: float = 1e-3
  """The learning rate."""
  schedule: Literal["adaptive", "fixed"] = "adaptive"
  """The learning rate schedule."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for Generalized Advantage Estimation (GAE)."""
  entropy_coef: float = 0.005
  """The coefficient for the entropy loss."""
  desired_kl: float = 0.01
  """The desired KL divergence between the new and old policies."""
  max_grad_norm: float = 1.0
  """The maximum gradient norm for the policy."""
  value_loss_coef: float = 1.0
  """The coefficient for the value loss."""
  use_clipped_value_loss: bool = True
  """Whether to use clipped value loss."""
  clip_param: float = 0.2
  """The clipping parameter for the policy."""
  normalize_advantage_per_mini_batch: bool = False
  """Whether to normalize the advantage per mini-batch. Default is False. If True, the
  advantage is normalized over the mini-batches only. Otherwise, the advantage is
  normalized over the entire collected trajectories.
  """
  class_name: str = "PPO"
  """Ignore, required by RSL-RL."""


@dataclass
class RslRlReppoAlgorithmCfg:
  """Config for the official REPPO algorithm."""

  num_learning_epochs: int = 4
  """The number of learning epochs per update."""
  num_mini_batches: int = 32
  """The number of mini-batches per update."""
  learning_rate: float = 3e-4
  """The learning rate."""
  gamma: float = 0.99
  """The discount factor."""
  lam: float = 0.95
  """The lambda parameter for return recursion."""
  max_grad_norm: float = 0.5
  """The maximum gradient norm."""
  desired_kl: float = 0.1
  """The KL bound used by REPPO."""
  target_entropy: float = -0.5
  """Entropy target multiplier per action dimension used by REPPO."""
  clip_param: float = 0.2
  """PPO-style ratio clipping parameter used by REPPO hybrid/PPO-only actor routes."""
  actor_route: Literal["reppo", "hybrid", "ppo_only"] = "reppo"
  """Actor objective route: original REPPO, cosine-gated hybrid, or PPO-only with ActorQ/HL-Gauss."""
  ppo_advantage_normalization: bool = False
  """Whether to normalize PPO-style advantages per mini-batch instead of once per rollout."""
  ppo_entropy_coef: float = 0.01
  """Fixed entropy coefficient used by PPO-style hybrid/PPO-only update routes."""
  ppo_value_loss_coef: float = 1.0
  """Value-loss coefficient used by PPO-style hybrid/PPO-only update routes."""
  ppo_schedule: Literal["adaptive", "fixed"] = "adaptive"
  """Learning-rate schedule used by the PPO-only actor route."""
  cosine_weight_min: float = 0.0
  """Minimum REPPO weight for the hybrid actor route."""
  cosine_weight_max: float = 1.0
  """Maximum REPPO weight for the hybrid actor route."""
  cosine_weight_power: float = 1.0
  """Power applied to clamped positive cosine before mapping it to the REPPO hybrid weight."""
  rnd_cfg: dict | None = None
  """Optional RND configuration."""
  symmetry_cfg: dict | None = None
  """Optional symmetry augmentation configuration."""
  scale_actions: bool = False
  """Whether to rescale actions before stepping the environment."""
  class_name: str = "REPPO"
  """Official REPPO algorithm class name."""


@dataclass
class RslRlBaseRunnerCfg:
  seed: int = 42
  """The seed for the experiment. Default is 42."""
  num_steps_per_env: int = 24
  """The number of steps per environment update."""
  max_iterations: int = 300
  """The maximum number of iterations."""
  obs_groups: dict[str, tuple[str, ...]] = field(
    default_factory=lambda: {"policy": ("policy",), "critic": ("critic",)},
  )
  save_interval: int = 50
  """The number of iterations between saves."""
  experiment_name: str = "exp1"
  """The experiment name."""
  run_name: str = ""
  """The run name. Default is empty string."""
  logger: Literal["wandb", "tensorboard"] = "wandb"
  """The logger to use. Default is wandb."""
  wandb_project: str = "mjlab"
  """The wandb project name."""
  wandb_tags: Tuple[str, ...] = ()
  """Tags for the wandb run. Default is empty tuple."""
  resume: bool = False
  """Whether to resume the experiment. Default is False."""
  load_run: str = ".*"
  """The run directory to load. Default is ".*" which means all runs. If regex
  expression, the latest (alphabetical order) matching run will be loaded.
  """
  load_checkpoint: str = "model_.*.pt"
  """The checkpoint file to load. Default is "model_.*.pt" (all). If regex expression,
  the latest (alphabetical order) matching file will be loaded.
  """
  clip_actions: float | None = None
  """The clipping range for action values. If None (default), no clipping is applied."""


@dataclass
class RslRlOnPolicyRunnerCfg(RslRlBaseRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. Default is OnPolicyRunner."""
  policy: RslRlPpoActorCriticCfg = field(default_factory=RslRlPpoActorCriticCfg)
  """The policy configuration."""
  algorithm: RslRlPpoAlgorithmCfg = field(default_factory=RslRlPpoAlgorithmCfg)
  """The algorithm configuration."""


@dataclass
class RslRlReppoRunnerCfg(RslRlOnPolicyRunnerCfg):
  class_name: str = "OnPolicyRunner"
  """The runner class name. REPPO now runs through the on-policy runner path."""
  policy: RslRlReppoActorCriticCfg = field(default_factory=RslRlReppoActorCriticCfg)
  """The REPPO policy configuration."""
  algorithm: RslRlReppoAlgorithmCfg = field(default_factory=RslRlReppoAlgorithmCfg)
  """The REPPO algorithm configuration."""


@dataclass
class RslRlFastTd3ActorCriticCfg(RslRlPpoActorCriticCfg):
  """Config for the FastTD3 actor and critic networks."""

  actor_hidden_dims: Tuple[int, ...] = (512, 256, 128)
  """The hidden dimensions of the actor network."""
  critic_hidden_dims: Tuple[int, ...] = (1024, 512, 256)
  """The hidden dimensions of the critic network."""
  actor_obs_normalization: bool = True
  """Whether to normalize actor observations."""
  critic_obs_normalization: bool = True
  """Whether to normalize critic observations."""
  init_scale: float = 0.01
  """Initialization scale for the actor head."""
  std_min: float = 0.05
  """Minimum exploration noise scale."""
  std_max: float = 0.8
  """Maximum exploration noise scale."""
  activation: str = "relu"
  """Activation function used by the actor/critic MLPs."""
  class_name: str = "FastTD3Actor"
  """FastTD3 actor class name."""
  critic_class_name: str = "FastTD3Critic"
  """FastTD3 critic class name."""


@dataclass
class RslRlFastTd3AlgorithmCfg(RslRlPpoAlgorithmCfg):
  """Config for the FastTD3 algorithm."""

  replay_size: int = 100_000
  """Replay buffer capacity."""
  batch_size: int = 256
  """Batch size sampled from replay."""
  learning_starts: int = 1_000
  """Number of environment steps before updates begin."""
  num_updates: int = 1
  """Number of gradient updates per iteration."""
  tau: float = 0.005
  """Target-network update coefficient."""
  target_noise: float = 0.2
  """Noise added to target actions during critic updates."""
  noise_clip: float = 0.5
  """Maximum absolute target noise."""
  policy_frequency: int = 2
  """Number of critic updates per actor update."""
  actor_learning_rate: float = 3e-4
  """Learning rate for the actor."""
  actor_learning_rate_end: float = 3e-4
  """Final cosine-annealed learning rate for the actor."""
  critic_learning_rate: float = 3e-4
  """Learning rate for the critic."""
  critic_learning_rate_end: float = 3e-4
  """Final cosine-annealed learning rate for the critic."""
  weight_decay: float = 0.1
  """Weight decay used by AdamW."""
  n_steps: int = 1
  """Number of replay steps to aggregate when sampling."""
  num_atoms: int = 101
  """Number of atoms in the critic support."""
  v_min: float = -250.0
  """Minimum critic support value."""
  v_max: float = 250.0
  """Maximum critic support value."""
  use_cdq: bool = True
  """Whether to use clipped double Q selection for actor and target updates."""
  reward_normalization: bool = False
  """Whether to normalize rewards with the running return scale."""
  optimizer: str = "adamw"
  """Optimizer used for actor and critic updates."""
  class_name: str = "FastTD3"
  """FastTD3 algorithm class name."""


@dataclass
class RslRlFastTd3RunnerCfg(RslRlOnPolicyRunnerCfg):
  class_name: str = "FastTD3Runner"
  """The runner class name. Default is FastTD3Runner."""
  policy: RslRlFastTd3ActorCriticCfg = field(default_factory=RslRlFastTd3ActorCriticCfg)
  """The FastTD3 policy configuration."""
  algorithm: RslRlFastTd3AlgorithmCfg = field(default_factory=RslRlFastTd3AlgorithmCfg)
  """The FastTD3 algorithm configuration."""
