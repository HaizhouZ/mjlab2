from pathlib import Path

from mjlab.utils.wandb import get_wandb_motion_cache_dir


def test_get_wandb_motion_cache_dir_prefers_explicit_env(
  monkeypatch,
) -> None:
  monkeypatch.setenv("MJLAB_WANDB_CACHE_DIR", "/tmp/mjlab-motion-cache")
  monkeypatch.delenv("MJLAB_CACHE_DIR", raising=False)
  monkeypatch.delenv("XDG_CACHE_HOME", raising=False)

  motions_dir = get_wandb_motion_cache_dir("entity", "project")

  assert motions_dir == Path("/tmp/mjlab-motion-cache/project/motions")


def test_get_wandb_motion_cache_dir_uses_xdg_cache_home(
  monkeypatch,
) -> None:
  monkeypatch.delenv("MJLAB_WANDB_CACHE_DIR", raising=False)
  monkeypatch.delenv("MJLAB_CACHE_DIR", raising=False)
  monkeypatch.setenv("XDG_CACHE_HOME", "/tmp/xdg-cache")

  motions_dir = get_wandb_motion_cache_dir("entity", "project")

  assert motions_dir == Path("/tmp/xdg-cache/mjlab/wandb_motions/project/motions")
