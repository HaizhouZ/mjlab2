"""RL configuration for Unitree G1 velocity task."""

from mjlab.rl import (
  RslRlOnPolicyRunnerCfg,
  RslRlPpoActorCriticCfg,
  RslRlPpoAlgorithmCfg,
  RslRlReppoActorCriticCfg,
  RslRlReppoAlgorithmCfg,
  RslRlReppoRunnerCfg,
)


def unitree_g1_ppo_runner_cfg() -> RslRlOnPolicyRunnerCfg:
  """Create RL runner configuration for Unitree G1 velocity task."""
  return RslRlOnPolicyRunnerCfg(
    policy=RslRlPpoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
    ),
    algorithm=RslRlPpoAlgorithmCfg(
      value_loss_coef=1.0,
      use_clipped_value_loss=True,
      clip_param=0.2,
      entropy_coef=0.01,
      num_learning_epochs=5,
      num_mini_batches=4,
      learning_rate=1.0e-3,
      schedule="adaptive",
      gamma=0.99,
      lam=0.95,
      desired_kl=0.01,
      max_grad_norm=1.0,
    ),
    experiment_name="g1_velocity",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )


def unitree_g1_reppo_runner_cfg() -> RslRlReppoRunnerCfg:
  """Create REPPO runner configuration for Unitree G1 flat velocity."""
  return RslRlReppoRunnerCfg(
    policy=RslRlReppoActorCriticCfg(
      init_noise_std=1.0,
      actor_obs_normalization=True,
      critic_obs_normalization=True,
      actor_hidden_dims=(512, 256, 128),
      critic_hidden_dims=(512, 256, 128),
      activation="elu",
      ent_start=0.001,
      kl_start=0.01,
      actor_min_std=0.1,
    ),
    algorithm=RslRlReppoAlgorithmCfg(
      learning_rate=3.0e-4,
      schedule="fixed",
      gamma=0.99,
      lam=0.95,
      max_grad_norm=0.5,
      num_learning_epochs=4,
      num_mini_batches=32,
      num_atoms=151,
      vmin=-20.0,
      vmax=15.0,
      aux_loss_mult=0.0,
      kl_bound=0.1,
      actor_kl_clip_mode="full",
      ent_target_mult=0.5,
    ),
    experiment_name="g1_velocity_reppo",
    save_interval=50,
    num_steps_per_env=24,
    max_iterations=30_000,
  )
