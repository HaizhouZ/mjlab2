from mjlab.tasks.registry import register_mjlab_task
from mjlab.rl import MjlabFastTD3Runner
from rsl_rl.runners import ReppoRunner

from .env_cfgs import yam_lift_cube_env_cfg
from .rl_cfg import yam_lift_cube_fasttd3_runner_cfg
from .rl_cfg import yam_lift_cube_reppo_runner_cfg

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Yam",
  env_cfg=yam_lift_cube_env_cfg(),
  play_env_cfg=yam_lift_cube_env_cfg(play=True),
  rl_cfg=yam_lift_cube_reppo_runner_cfg(),
  runner_cls=ReppoRunner,
)

register_mjlab_task(
  task_id="Mjlab-Lift-Cube-Yam-FastTD3",
  env_cfg=yam_lift_cube_env_cfg(),
  play_env_cfg=yam_lift_cube_env_cfg(play=True),
  rl_cfg=yam_lift_cube_fasttd3_runner_cfg(),
  runner_cls=MjlabFastTD3Runner,
)
