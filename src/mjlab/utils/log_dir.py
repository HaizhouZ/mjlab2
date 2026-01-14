import os
from datetime import datetime
from pathlib import Path


def get_log_dir(
  experiment_name: str, run_name: str | None = None, parent_dir: str = "rsl_rl"
) -> Path:
  env_log_dir: str | None = os.environ.get("MJLAB_LOG_DIR", None)
  log_root_path: Path
  if env_log_dir is not None:
    print(f"Using log directory from MJLAB_LOG_DIR: {env_log_dir}")
    log_root_path = Path(env_log_dir)
  else:
    print("[INFO] MJLAB_LOG_DIR not set, using default log directory.")
    log_root_path = Path("logs") / parent_dir

  log_root_path.resolve()
  log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  if run_name:
    log_file_name += f"_{run_name}"

  log_dir = log_root_path / experiment_name / log_file_name

  return log_dir
