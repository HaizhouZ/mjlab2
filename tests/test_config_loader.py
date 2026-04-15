"""Tests for the OmegaConf-backed config loader helpers."""

import json
from dataclasses import dataclass, field

import pytest
import yaml

from mjlab.utils.config_loader import (
    apply_config_overrides,
    load_dataclass_from_dict,
    load_dataclass_from_json,
    load_dataclass_from_yaml,
    merge_configs,
    override_dataclass_from_source,
    save_dataclass_to_json,
    save_dataclass_to_yaml,
)


@dataclass(kw_only=True)
class SimpleTestConfig:
    """Simple config for testing."""

    param1: int = 10
    param2: str = "default"
    param3: float = 3.14


@dataclass(kw_only=True)
class NestedConfig:
    """Nested config for testing."""

    simple: SimpleTestConfig = field(default_factory=SimpleTestConfig)
    param4: bool = True


class TestConfigLoading:
    """Test dataclass config loading helpers."""

    def test_load_from_yaml_basic(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            """
param1: 42
param2: "test"
param3: 2.71
"""
        )

        cfg = load_dataclass_from_yaml(SimpleTestConfig, yaml_file)
        assert cfg.param1 == 42
        assert cfg.param2 == "test"
        assert cfg.param3 == 2.71

    def test_load_from_yaml_partial(self, tmp_path):
        yaml_file = tmp_path / "config.yaml"
        yaml_file.write_text(
            """
param1: 100
"""
        )

        cfg = load_dataclass_from_yaml(SimpleTestConfig, yaml_file)
        assert cfg.param1 == 100
        assert cfg.param2 == "default"
        assert cfg.param3 == 3.14

    def test_load_from_yaml_empty(self, tmp_path):
        yaml_file = tmp_path / "empty.yaml"
        yaml_file.write_text("")

        cfg = load_dataclass_from_yaml(SimpleTestConfig, yaml_file)
        assert cfg.param1 == 10
        assert cfg.param2 == "default"
        assert cfg.param3 == 3.14

    def test_load_from_yaml_file_not_found(self, tmp_path):
        yaml_file = tmp_path / "nonexistent.yaml"
        with pytest.raises(FileNotFoundError):
            load_dataclass_from_yaml(SimpleTestConfig, yaml_file)

    def test_load_from_dict(self):
        data = {"param1": 50, "param2": "dict_value"}
        cfg = load_dataclass_from_dict(SimpleTestConfig, data)
        assert cfg.param1 == 50
        assert cfg.param2 == "dict_value"
        assert cfg.param3 == 3.14

    def test_load_from_dict_empty(self):
        cfg = load_dataclass_from_dict(SimpleTestConfig, {})
        assert cfg.param1 == 10
        assert cfg.param2 == "default"
        assert cfg.param3 == 3.14

    def test_load_dataclass_from_json(self, tmp_path):
        json_file = tmp_path / "config.json"
        json_file.write_text(json.dumps({"param1": 99, "param2": "json_test"}))

        cfg = load_dataclass_from_json(SimpleTestConfig, json_file)
        assert cfg.param1 == 99
        assert cfg.param2 == "json_test"
        assert cfg.param3 == 3.14

    def test_save_dataclass_to_yaml(self, tmp_path):
        yaml_file = tmp_path / "output.yaml"
        cfg = SimpleTestConfig(param1=77, param2="saved")
        save_dataclass_to_yaml(cfg, yaml_file)

        assert yaml_file.exists()
        with open(yaml_file) as f:
            data = yaml.safe_load(f)
        assert data["param1"] == 77
        assert data["param2"] == "saved"

    def test_save_dataclass_to_json(self, tmp_path):
        json_file = tmp_path / "output.json"
        cfg = SimpleTestConfig(param1=88, param2="json_saved")
        save_dataclass_to_json(cfg, json_file)

        assert json_file.exists()
        with open(json_file) as f:
            data = json.load(f)
        assert data["param1"] == 88
        assert data["param2"] == "json_saved"

    def test_nested_dataclass_load_from_yaml(self, tmp_path):
        yaml_file = tmp_path / "nested.yaml"
        yaml_file.write_text(
            """
param4: false
simple:
  param1: 200
  param2: "nested"
  param3: 1.5
"""
        )

        cfg = load_dataclass_from_yaml(NestedConfig, yaml_file)
        assert cfg.param4 is False
        assert cfg.simple.param1 == 200
        assert cfg.simple.param2 == "nested"
        assert cfg.simple.param3 == 1.5

    def test_nested_config_save_load_roundtrip(self, tmp_path):
        cfg = NestedConfig(
            simple=SimpleTestConfig(param1=123, param2="test"),
            param4=False,
        )

        yaml_file = tmp_path / "roundtrip.yaml"
        save_dataclass_to_yaml(cfg, yaml_file)

        loaded_cfg = load_dataclass_from_yaml(NestedConfig, yaml_file)
        assert loaded_cfg.simple.param1 == 123
        assert loaded_cfg.simple.param2 == "test"
        assert loaded_cfg.param4 is False

    def test_load_from_dict_with_extra_fields(self):
        data = {"param1": 50, "param2": "test", "extra_field": "ignored"}
        cfg = load_dataclass_from_dict(SimpleTestConfig, data)
        assert cfg.param1 == 50
        assert cfg.param2 == "test"

    def test_override_from_source(self, tmp_path):
        yaml_file = tmp_path / "override.yaml"
        yaml_file.write_text("param2: changed\n")

        cfg = SimpleTestConfig(param1=5)
        override_dataclass_from_source(cfg, yaml_file)
        assert cfg.param1 == 5
        assert cfg.param2 == "changed"


class TestApplyConfigOverrides:
    """Test apply_config_overrides function."""

    def test_apply_simple_overrides(self):
        cfg = SimpleTestConfig()
        overrides = {"param1": 999, "param2": "override"}
        apply_config_overrides(cfg, overrides)

        assert cfg.param1 == 999
        assert cfg.param2 == "override"
        assert cfg.param3 == 3.14

    def test_apply_overrides_nonexistent_field(self):
        cfg = SimpleTestConfig()
        overrides = {"nonexistent": 42}
        with pytest.raises(AttributeError):
            apply_config_overrides(cfg, overrides)

    def test_apply_nested_overrides(self):
        cfg = NestedConfig()
        overrides = {"simple": {"param1": 555, "param2": "nested_override"}}
        apply_config_overrides(cfg, overrides)

        assert cfg.simple.param1 == 555
        assert cfg.simple.param2 == "nested_override"
        assert cfg.simple.param3 == 3.14

    def test_apply_empty_overrides(self):
        cfg = SimpleTestConfig(param1=42)
        apply_config_overrides(cfg, {})
        assert cfg.param1 == 42

    def test_apply_overrides_recursive_false(self):
        cfg = NestedConfig()
        new_simple = SimpleTestConfig(param1=777)
        overrides = {"simple": new_simple}
        apply_config_overrides(cfg, overrides, recursive=False)

        assert cfg.simple == new_simple
        assert cfg.simple.param1 == 777


class TestMergeConfigs:
    """Test merge_configs function."""

    def test_merge_configs_simple(self):
        base_cfg = SimpleTestConfig(param1=10, param2="base")
        override_cfg = SimpleTestConfig(param1=20, param2="override")

        result = merge_configs(base_cfg, override_cfg)

        assert result.param1 == 20
        assert result.param2 == "override"

    def test_merge_configs_nested(self):
        base_cfg = NestedConfig(
            simple=SimpleTestConfig(param1=10, param2="base"),
            param4=True,
        )
        override_cfg = NestedConfig(
            simple=SimpleTestConfig(param1=30),
            param4=False,
        )

        result = merge_configs(base_cfg, override_cfg)

        assert result.param4 is False
        assert result.simple.param1 == 30

    def test_merge_configs_roundtrip(self, tmp_path):
        base_cfg = SimpleTestConfig(param1=100, param2="base", param3=1.0)

        yaml_file = tmp_path / "base.yaml"
        save_dataclass_to_yaml(base_cfg, yaml_file)

        loaded_cfg = load_dataclass_from_yaml(SimpleTestConfig, yaml_file)
        assert loaded_cfg.param1 == 100
        assert loaded_cfg.param2 == "base"
        assert loaded_cfg.param3 == 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
