"""Tests for Tyro + OmegaConf CLI helpers."""

from dataclasses import dataclass, field

import pytest

from mjlab.utils.omegaconf_cli import parse_choice_and_dataclass, parse_dataclass_cli


@dataclass(kw_only=True)
class InnerCliCfg:
    count: int = 4
    enabled: bool = True


@dataclass(kw_only=True)
class OuterCliCfg:
    name: str = "demo"
    inner: InnerCliCfg = field(default_factory=InnerCliCfg)


@dataclass(kw_only=True)
class BaseEnvCfg:
    num_envs: int = 1


@dataclass(kw_only=True)
class TrackingEnvCfg(BaseEnvCfg):
    motion_horizon: int = 2


@dataclass(kw_only=True)
class DynamicTrainCfg:
    env: BaseEnvCfg = field(default_factory=BaseEnvCfg)
    name: str = "demo"


def test_parse_dataclass_cli_supports_current_override_style(tmp_path):
    yaml_path = tmp_path / "cfg.yaml"
    yaml_path.write_text("name: from_file\ninner:\n  count: 2\n")

    cfg = parse_dataclass_cli(
        OuterCliCfg,
        argv=["config_path=" + str(yaml_path), "inner.count=9", "--inner.enabled", "false"],
    )

    assert cfg.name == "from_file"
    assert cfg.inner.count == 9
    assert cfg.inner.enabled is False


def test_parse_dataclass_cli_help_is_tyro_generated(capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_dataclass_cli(OuterCliCfg, argv=["--help"])

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "--config-path" in output
    assert "--config-source" in output
    assert "--name" in output
    assert "--inner.count" in output


def test_parse_choice_and_dataclass_help_shows_choices(capsys):
    with pytest.raises(SystemExit) as excinfo:
        parse_choice_and_dataclass(
            ["task_a", "task_b"], OuterCliCfg, argv=["--help"], prog="play"
        )

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "Usage: play <choice>" in output
    assert "  task_a" in output
    assert "Run `play <choice> --help`" in output


def test_parse_dataclass_cli_uses_concrete_default_type_for_nested_help(capsys):
    default = DynamicTrainCfg(env=TrackingEnvCfg())

    with pytest.raises(SystemExit) as excinfo:
        parse_dataclass_cli(DynamicTrainCfg, argv=["--help"], default=default)

    assert excinfo.value.code == 0
    output = capsys.readouterr().out
    assert "--env.num-envs" in output
    assert "--env.motion-horizon" in output


def test_parse_dataclass_cli_parses_concrete_nested_default_fields():
    default = DynamicTrainCfg(env=TrackingEnvCfg())

    cfg = parse_dataclass_cli(
        DynamicTrainCfg,
        argv=["env.motion_horizon=7", "env.num_envs=16"],
        default=default,
    )

    assert isinstance(cfg.env, TrackingEnvCfg)
    assert cfg.env.motion_horizon == 7
    assert cfg.env.num_envs == 16
