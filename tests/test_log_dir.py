from pathlib import Path

from mjlab.utils.log_dir import get_log_dir


def test_get_log_dir_prefers_explicit_log_path(monkeypatch) -> None:
  monkeypatch.setenv("MJLAB_LOG_PATH", "/tmp/mjlab-logs")
  monkeypatch.setenv("MJLAB_OUTPUT_DIR", "/tmp/mjlab-output")

  log_dir = get_log_dir("exp_name", run_name="run_name")

  assert log_dir.parent == Path("/tmp/mjlab-logs/exp_name")
  assert log_dir.name.endswith("_run_name")


def test_get_log_dir_uses_output_dir_when_present(monkeypatch) -> None:
  monkeypatch.delenv("MJLAB_LOG_PATH", raising=False)
  monkeypatch.setenv("MJLAB_OUTPUT_DIR", "/tmp/mjlab-output")

  log_dir = get_log_dir("exp_name", parent_dir="runner_logs")

  assert log_dir.parent == Path("/tmp/mjlab-output/runner_logs/exp_name")
