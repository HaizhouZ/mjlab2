from types import SimpleNamespace

from mjlab.tasks.tracking.rl.runner import _save_tracking_policy
from mjlab.tasks.velocity.rl.runner import _export_velocity_policy


class _DummyPolicy:
  actor_obs_normalization = False


def test_save_tracking_policy_uses_logger_logger_type(monkeypatch) -> None:
  calls: dict[str, object] = {}

  def fake_export(env, policy, normalizer, path, filename):
    calls["export"] = {
      "env": env,
      "policy": policy,
      "normalizer": normalizer,
      "path": path,
      "filename": filename,
    }

  def fake_attach(env, run_name, path, filename):
    calls["attach"] = {
      "env": env,
      "run_name": run_name,
      "path": path,
      "filename": filename,
    }

  def fake_save(path, base_path):
    calls["wandb_save"] = {"path": path, "base_path": base_path}

  import mjlab.tasks.tracking.rl.runner as tracking_runner

  monkeypatch.setattr(tracking_runner, "export_motion_policy_as_onnx", fake_export)
  monkeypatch.setattr(tracking_runner, "attach_onnx_metadata", fake_attach)
  monkeypatch.setattr(tracking_runner.wandb, "save", fake_save)
  monkeypatch.setattr(
    tracking_runner.wandb,
    "run",
    SimpleNamespace(name="tracking-run", use_artifact=lambda _name: None),
  )

  runner = SimpleNamespace(
    logger=SimpleNamespace(logger_type="wandb"),
    alg=SimpleNamespace(policy=_DummyPolicy()),
    env=SimpleNamespace(unwrapped=object()),
    registry_name=None,
  )

  _save_tracking_policy(runner, "/tmp/run_dir/model_10.pt")

  assert calls["attach"]["run_name"] == "tracking-run"
  assert calls["wandb_save"]["path"] == "/tmp/run_dir/run_dir.onnx"


def test_export_velocity_policy_uses_logger_logger_type(monkeypatch) -> None:
  calls: dict[str, object] = {}

  def fake_export(policy, normalizer, path, filename):
    calls["export"] = {
      "policy": policy,
      "normalizer": normalizer,
      "path": path,
      "filename": filename,
    }

  def fake_attach(env, run_name, path, filename):
    calls["attach"] = {
      "env": env,
      "run_name": run_name,
      "path": path,
      "filename": filename,
    }

  def fake_save(path, base_path):
    calls["wandb_save"] = {"path": path, "base_path": base_path}

  import mjlab.tasks.velocity.rl.runner as velocity_runner

  monkeypatch.setattr(velocity_runner, "export_velocity_policy_as_onnx", fake_export)
  monkeypatch.setattr(velocity_runner, "attach_onnx_metadata", fake_attach)
  monkeypatch.setattr(velocity_runner.wandb, "save", fake_save)
  monkeypatch.setattr(velocity_runner.wandb, "run", SimpleNamespace(name="velocity-run"))

  runner = SimpleNamespace(
    logger=SimpleNamespace(logger_type="wandb"),
    alg=SimpleNamespace(policy=_DummyPolicy()),
    env=SimpleNamespace(unwrapped=object()),
  )

  _export_velocity_policy(runner, "/tmp/velocity_run/model_50.pt")

  assert calls["attach"]["run_name"] == "velocity-run"
  assert calls["wandb_save"]["path"] == "/tmp/velocity_run/velocity_run.onnx"
