from mjlab.rl import RslRlFastTd3RunnerCfg
from mjlab.rl import RslRlReppoRunnerCfg


def yam_lift_cube_reppo_runner_cfg() -> RslRlReppoRunnerCfg:
  cfg = RslRlReppoRunnerCfg(
    experiment_name="yam_lift_cube",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )
  cfg.policy.init_noise_std = 1.0
  cfg.policy.ent_start = 0.001
  cfg.policy.kl_start = 0.01
  cfg.policy.actor_min_std = 0.1
  cfg.policy.actor_obs_normalization = True
  cfg.policy.critic_obs_normalization = True
  cfg.policy.actor_hidden_dims = (512, 256, 128)
  cfg.policy.critic_hidden_dims = (512, 256, 128)
  cfg.policy.activation = "elu"
  cfg.algorithm.num_learning_epochs = 4
  cfg.algorithm.num_mini_batches = 32
  cfg.algorithm.learning_rate = 3.0e-4
  cfg.algorithm.schedule = "fixed"
  cfg.algorithm.gamma = 0.99
  cfg.algorithm.lam = 0.95
  cfg.algorithm.max_grad_norm = 0.5
  cfg.algorithm.num_atoms = 151
  cfg.algorithm.vmin = 0.0
  cfg.algorithm.vmax = 150.0
  cfg.algorithm.aux_loss_mult = 0.0
  cfg.algorithm.kl_bound = 0.1
  cfg.algorithm.actor_kl_clip_mode = "full"
  cfg.algorithm.ent_target_mult = 0.5
  cfg.algorithm.num_action_samples = 64
  return cfg


def yam_lift_cube_ppo_runner_cfg() -> RslRlReppoRunnerCfg:
  return yam_lift_cube_reppo_runner_cfg()


def yam_lift_cube_fasttd3_runner_cfg() -> RslRlFastTd3RunnerCfg:
  cfg = RslRlFastTd3RunnerCfg(
    experiment_name="yam_lift_cube_fasttd3",
    save_interval=100,
    num_steps_per_env=24,
    max_iterations=5_000,
  )
  cfg.policy.actor_obs_normalization = True
  cfg.policy.critic_obs_normalization = True
  cfg.policy.actor_hidden_dims = (512, 256, 128)
  cfg.policy.critic_hidden_dims = (1024, 512, 256)
  cfg.policy.init_scale = 0.01
  cfg.policy.std_min = 0.05
  cfg.policy.std_max = 0.8
  cfg.policy.activation = "relu"
  cfg.algorithm.replay_size = 50_000
  cfg.algorithm.batch_size = 128
  cfg.algorithm.learning_starts = 1_000
  cfg.algorithm.num_updates = 1
  cfg.algorithm.max_grad_norm = 0.5
  cfg.algorithm.tau = 0.005
  cfg.algorithm.policy_frequency = 2
  cfg.algorithm.target_noise = 0.2
  cfg.algorithm.noise_clip = 0.5
  cfg.algorithm.optimizer = "adamw"
  cfg.algorithm.actor_learning_rate = 3.0e-4
  cfg.algorithm.actor_learning_rate_end = 3.0e-4
  cfg.algorithm.critic_learning_rate = 3.0e-4
  cfg.algorithm.critic_learning_rate_end = 3.0e-4
  cfg.algorithm.weight_decay = 0.1
  cfg.algorithm.n_steps = 1
  cfg.algorithm.num_atoms = 101
  cfg.algorithm.v_min = -250.0
  cfg.algorithm.v_max = 250.0
  cfg.algorithm.use_cdq = True
  return cfg
