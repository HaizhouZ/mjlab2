"""OmegaConf-backed helpers for runtime configuration loading and overrides."""

from __future__ import annotations

import copy
import dataclasses
import json
import types
import urllib.request
from collections.abc import Mapping
from pathlib import Path
from typing import Any, TypeVar, get_args, get_origin

import yaml
from omegaconf import DictConfig, ListConfig, OmegaConf

T = TypeVar("T")


def _config_to_data(value: Any, *, resolve: bool = False) -> Any:
    """Convert dataclasses and OmegaConf containers into plain Python data."""
    if isinstance(value, (DictConfig, ListConfig)):
        return OmegaConf.to_container(value, resolve=resolve)
    if dataclasses.is_dataclass(value):
        result = {}
        for field in dataclasses.fields(value):
            result[field.name] = _config_to_data(
                getattr(value, field.name), resolve=resolve
            )
        return result
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {
            key: _config_to_data(item, resolve=resolve) for key, item in value.items()
        }
    if isinstance(value, list):
        return [_config_to_data(item, resolve=resolve) for item in value]
    if isinstance(value, tuple):
        return tuple(_config_to_data(item, resolve=resolve) for item in value)
    return value


def _ensure_mapping(data: Any, source_name: str) -> dict[str, Any]:
    if data is None:
        return {}
    if isinstance(data, (DictConfig, ListConfig)):
        data = OmegaConf.to_container(data, resolve=True)
    if not isinstance(data, dict):
        raise ValueError(f"{source_name} must contain a mapping, got {type(data)}")
    return data


def _set_attr(obj: Any, key: str, value: Any) -> None:
    try:
        setattr(obj, key, value)
    except (AttributeError, dataclasses.FrozenInstanceError):
        object.__setattr__(obj, key, value)


def _coerce_like(current_value: Any, value: Any) -> Any:
    if isinstance(current_value, Path) and isinstance(value, str):
        return Path(value)
    if isinstance(current_value, tuple) and isinstance(value, list):
        return tuple(value)
    if isinstance(current_value, list) and isinstance(value, tuple):
        return list(value)
    return value


def load_config_path(filepath: str | Path) -> dict[str, Any]:
    """Load a local YAML or JSON config file into a mapping."""
    filepath = Path(filepath)
    if not filepath.exists():
        raise FileNotFoundError(f"Config file not found: {filepath}")

    if filepath.suffix in {".yaml", ".yml"}:
        data = OmegaConf.to_container(OmegaConf.load(filepath), resolve=True)
    elif filepath.suffix == ".json":
        with open(filepath, "r", encoding="utf-8") as f:
            data = json.load(f)
    else:
        raise ValueError(f"Unsupported config file format: {filepath.suffix}")

    return _ensure_mapping(data, str(filepath))


def load_config_source(source: str) -> dict[str, Any]:
    """Load config data from a local path, HTTP(S) URL, or W&B artifact URI."""
    if source.startswith(("http://", "https://")):
        with urllib.request.urlopen(source) as response:
            payload = response.read().decode("utf-8")
        data = yaml.safe_load(payload)
        return _ensure_mapping(data, source)

    if source.startswith("wandb://"):
        try:
            import wandb
        except ImportError as e:
            raise ImportError("wandb is required for wandb:// config sources") from e

        raw = source[len("wandb://") :]
        parts = raw.split("/", 3)
        if len(parts) < 3:
            raise ValueError(
                "Invalid wandb config source. Expected "
                "wandb://<entity>/<project>/<artifact[:alias]>/<config_path>"
            )

        entity, project, artifact_ref = parts[0], parts[1], parts[2]
        config_rel_path = parts[3] if len(parts) == 4 and parts[3] else "config.yaml"
        if ":" not in artifact_ref:
            artifact_ref += ":latest"

        api = wandb.Api()
        artifact = api.artifact(f"{entity}/{project}/{artifact_ref}")
        config_file = Path(artifact.download()) / config_rel_path
        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file '{config_rel_path}' not found in W&B artifact '{artifact_ref}'."
            )
        return load_config_path(config_file)

    return load_config_path(source)


def apply_config_overrides(
    cfg: T, overrides: dict[str, Any] | DictConfig, recursive: bool = True
) -> T:
    """Apply mapping-style overrides to an object graph in place."""
    overrides_data = _ensure_mapping(overrides, "overrides")
    for key, value in overrides_data.items():
        if dataclasses.is_dataclass(cfg):
            valid_fields = {field.name for field in dataclasses.fields(cfg)}
            if key not in valid_fields:
                raise AttributeError(
                    f"Configuration {type(cfg).__name__} has no field '{key}'"
                )
        elif not hasattr(cfg, key):
            raise AttributeError(
                f"Configuration {type(cfg).__name__} has no field '{key}'"
            )

        current_value = getattr(cfg, key)
        if recursive and dataclasses.is_dataclass(current_value) and isinstance(
            value, Mapping
        ):
            apply_config_overrides(current_value, dict(value), recursive=True)
            continue
        if recursive and isinstance(current_value, dict) and isinstance(value, Mapping):
            for subkey, subvalue in value.items():
                if (
                    subkey in current_value
                    and dataclasses.is_dataclass(current_value[subkey])
                    and isinstance(subvalue, Mapping)
                ):
                    apply_config_overrides(
                        current_value[subkey], dict(subvalue), recursive=True
                    )
                elif (
                    subkey in current_value
                    and isinstance(current_value[subkey], dict)
                    and isinstance(subvalue, Mapping)
                ):
                    current_value[subkey].update(dict(subvalue))
                else:
                    current_value[subkey] = subvalue
            continue
        if recursive and isinstance(current_value, tuple) and isinstance(value, (list, tuple)):
            updated_items = list(current_value)
            for index, subvalue in enumerate(value):
                if index >= len(updated_items):
                    updated_items.append(subvalue)
                    continue
                current_item = updated_items[index]
                if dataclasses.is_dataclass(current_item) and isinstance(subvalue, Mapping):
                    apply_config_overrides(current_item, dict(subvalue), recursive=True)
                elif isinstance(current_item, dict) and isinstance(subvalue, Mapping):
                    current_item.update(dict(subvalue))
                else:
                    updated_items[index] = _coerce_like(current_item, subvalue)
            _set_attr(cfg, key, tuple(updated_items))
            continue
        if recursive and isinstance(current_value, list) and isinstance(value, (list, tuple)):
            updated_items = list(current_value)
            for index, subvalue in enumerate(value):
                if index >= len(updated_items):
                    updated_items.append(subvalue)
                    continue
                current_item = updated_items[index]
                if dataclasses.is_dataclass(current_item) and isinstance(subvalue, Mapping):
                    apply_config_overrides(current_item, dict(subvalue), recursive=True)
                elif isinstance(current_item, dict) and isinstance(subvalue, Mapping):
                    current_item.update(dict(subvalue))
                else:
                    updated_items[index] = _coerce_like(current_item, subvalue)
            _set_attr(cfg, key, updated_items)
            continue
        _set_attr(cfg, key, _coerce_like(current_value, value))
    return cfg


def _extract_nondefault_overrides(value: Any, default_value: Any) -> Any:
    if isinstance(value, dict) and isinstance(default_value, dict):
        result = {}
        for key, item in value.items():
            nested_default = default_value.get(key)
            nested = _extract_nondefault_overrides(item, nested_default)
            if nested not in ({}, None):
                result[key] = nested
        return result
    if value != default_value:
        return value
    return None


def merge_configs(base_cfg: T, override_cfg: T, recursive: bool = True) -> T:
    """Merge non-default values from one config object onto another."""
    override_data = _config_to_data(override_cfg, resolve=True)
    try:
        default_cfg = type(override_cfg)()
        default_data = _config_to_data(default_cfg, resolve=True)
        overrides = _extract_nondefault_overrides(override_data, default_data) or {}
    except TypeError:
        overrides = override_data

    return apply_config_overrides(base_cfg, overrides, recursive=recursive)


def _coerce_scalar(annotation: Any, value: Any) -> Any:
    if annotation is Any:
        return value
    if annotation is bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, str):
            lowered = value.lower()
            if lowered in {"true", "1", "yes", "on"}:
                return True
            if lowered in {"false", "0", "no", "off"}:
                return False
        raise ValueError(f"Cannot coerce {value!r} to bool")
    if annotation in {int, float, str}:
        return annotation(value)
    if annotation is Path:
        return Path(value)
    return value


def _construct_value(annotation: Any, value: Any) -> Any:
    if value is None:
        return None
    if annotation is Any:
        return value

    origin = get_origin(annotation)
    if origin is None and isinstance(annotation, types.UnionType):
        origin = types.UnionType

    if origin is not None:
        if str(origin) == "typing.Union" or origin is types.UnionType:
            args = [arg for arg in get_args(annotation) if arg is not type(None)]
            for arg in args:
                try:
                    return _construct_value(arg, value)
                except Exception:
                    continue
            return value
        if str(origin).endswith("Literal"):
            return value
        if origin is list:
            elem_type = get_args(annotation)[0] if get_args(annotation) else Any
            return [_construct_value(elem_type, item) for item in value]
        if origin is tuple:
            args = get_args(annotation)
            if len(args) == 2 and args[1] is Ellipsis:
                return tuple(_construct_value(args[0], item) for item in value)
            return tuple(
                _construct_value(arg, item)
                for arg, item in zip(args, value, strict=False)
            )
        if origin is dict:
            key_type, value_type = (
                get_args(annotation) if get_args(annotation) else (Any, Any)
            )
            return {
                _construct_value(key_type, key): _construct_value(value_type, item)
                for key, item in value.items()
            }

    if (
        isinstance(annotation, type)
        and dataclasses.is_dataclass(annotation)
        and isinstance(value, Mapping)
    ):
        return load_dataclass_from_dict(annotation, dict(value))

    if isinstance(annotation, type):
        return _coerce_scalar(annotation, value)

    return value


def load_dataclass_from_dict(cls: type[T], data: dict[str, Any]) -> T:
    """Instantiate a dataclass from a plain mapping."""
    if not dataclasses.is_dataclass(cls):
        raise TypeError(f"{cls!r} is not a dataclass type")

    kwargs = {}
    for field in dataclasses.fields(cls):
        if field.name not in data:
            continue
        kwargs[field.name] = _construct_value(field.type, data[field.name])
    return cls(**kwargs)


def load_dataclass_from_yaml(cls: type[T], filepath: str | Path) -> T:
    return load_dataclass_from_dict(cls, load_config_path(filepath))


def load_dataclass_from_json(cls: type[T], filepath: str | Path) -> T:
    return load_dataclass_from_dict(cls, load_config_path(filepath))


def save_dataclass_to_yaml(cfg: Any, filepath: str | Path, sort_keys: bool = False) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        yaml.safe_dump(_config_to_data(cfg, resolve=True), f, sort_keys=sort_keys)


def save_dataclass_to_json(cfg: Any, filepath: str | Path, indent: int = 2) -> None:
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(_config_to_data(cfg, resolve=True), f, indent=indent)


def override_dataclass_from_source(cfg: T, source: str | Path) -> T:
    """Load a config source and apply only the provided fields onto cfg."""
    return apply_config_overrides(cfg, load_config_source(str(source)), recursive=True)
