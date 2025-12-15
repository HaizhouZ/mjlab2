from mjlab.tasks.registry import register_mjlab_task
from mjlab.tasks.tracking.rl import MotionTrackingOnPolicyRunner

from .env_cfgs import (
  unitree_g1_flat_multitracking_env_cfg_box,
  unitree_g1_flat_multitracking_env_cfg_largebox,
  unitree_g1_flat_tracking_env_cfg,
  unitree_g1_flat_tracking_env_cfg_box,
  unitree_g1_flat_tracking_env_cfg_largebox,
  unitree_g1_flat_tracking_env_cfg_largebox_pdtargets,
)
from .rl_cfg import (
  unitree_g1_tracking_ppo_runner_cfg,
  unitree_g1_tracking_ppo_runner_cfg_box,
  unitree_g1_tracking_ppo_runner_cfg_largebox,
  unitree_g1_tracking_ppo_runner_cfg_largebox_pdtargets,
  unitree_g1_tracking_ppo_runner_cfg_multitracking_box,
  unitree_g1_tracking_ppo_runner_cfg_multitracking_largebox,
)

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

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Box",
  env_cfg=unitree_g1_flat_tracking_env_cfg_box(),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg_box(play=True),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg_box(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-Box-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg_box(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg_box(
    has_state_estimation=False, play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg_box(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-MultiTracking-Flat-Unitree-G1-Box-No-State-Estimation",
  env_cfg=unitree_g1_flat_multitracking_env_cfg_box(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_multitracking_env_cfg_box(
    has_state_estimation=False, play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg_multitracking_box(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-LargeBox-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg_largebox(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg_largebox(
    has_state_estimation=False, play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg_largebox(),
  runner_cls=MotionTrackingOnPolicyRunner,
)


register_mjlab_task(
  task_id="Mjlab-MultiTracking-Flat-Unitree-G1-LargeBox-No-State-Estimation",
  env_cfg=unitree_g1_flat_multitracking_env_cfg_largebox(has_state_estimation=False),
  play_env_cfg=unitree_g1_flat_multitracking_env_cfg_largebox(
    has_state_estimation=False, play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg_multitracking_largebox(),
  runner_cls=MotionTrackingOnPolicyRunner,
)

register_mjlab_task(
  task_id="Mjlab-Tracking-Flat-Unitree-G1-LargeBox-PDTargets-No-State-Estimation",
  env_cfg=unitree_g1_flat_tracking_env_cfg_largebox_pdtargets(
    has_state_estimation=False
  ),
  play_env_cfg=unitree_g1_flat_tracking_env_cfg_largebox_pdtargets(
    has_state_estimation=False, play=True
  ),
  rl_cfg=unitree_g1_tracking_ppo_runner_cfg_largebox_pdtargets(),
  runner_cls=MotionTrackingOnPolicyRunner,
)
