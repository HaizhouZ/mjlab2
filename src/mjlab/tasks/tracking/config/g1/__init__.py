from mjlab.tasks.registry import register_mjlab_task  # noqa: I001
from mjlab.tasks.tracking.rl import (
  MotionTrackingOnPolicyRunner,
  MotionTrackingReppoRunner,
)

from .env_cfgs import (
  unitree_g1_flat_tracking_env_cfg,
)
from .multitracking_env_cfgs import (
  unitree_g1_flat_multitracking_env_cfg,
)
from .rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg,
  unitree_g1_multitracking_ppo_runner_cfg,
  unitree_g1_multitracking_reppo_runner_cfg,
)

################################################################################
# Robot Tracking Only Tasks
################################################################################

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_tracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg(has_state_estimation=False, play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

################################################################################
# Multi-Trajectory Tracking Tasks
################################################################################

register_mjlab_task(
  task_id="Mjlab-MultiTracking-Flat-Unitree-G1",
  env_cfg=unitree_g1_flat_multitracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_multitracking_env_cfg(play=True),
  rl_cfg=unitree_g1_multitracking_ppo_runner_cfg(),
  runner_cls=MotionTrackingOnPolicyRunner,
)


register_mjlab_task(
  task_id="Mjlab-MultiTracking-Flat-Unitree-G1-REPPO",
  env_cfg=unitree_g1_flat_multitracking_env_cfg(),
  play_env_cfg=unitree_g1_flat_multitracking_env_cfg(play=True),
  rl_cfg=unitree_g1_multitracking_reppo_runner_cfg(),
  runner_cls=MotionTrackingReppoRunner,
)
