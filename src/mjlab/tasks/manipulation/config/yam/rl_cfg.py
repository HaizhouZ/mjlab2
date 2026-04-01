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
