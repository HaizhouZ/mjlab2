"""Unified configuration loader for parsing configs from YAML, W&B, and other sources.

This module provides a flexible framework for loading dataclass configurations from
external sources like YAML files and Weights & Biases. It uses a flag-based system
to mark which dataclasses are readable from external config files.

Example:
    Mark a dataclass as config-loadable:
    
    >>> from mjlab.utils.config_loader import ConfigLoadable
    >>> from dataclasses import dataclass
    >>> 
    >>> @dataclass(kw_only=True)
    >>> class MyConfig(ConfigLoadable):
    ...     param1: int = 10
    ...     param2: str = "test"
    
    Load from YAML file:
    
    >>> cfg = MyConfig.load_from_yaml("config.yaml")
    
    Load from W&B:
    
    >>> cfg = MyConfig.load_from_wandb(
    ...     wandb_entity="my_entity",
    ...     wandb_project="my_project",
    ...     config_path="path/to/config.yaml",
    ... )
"""

import dataclasses
import json
from pathlib import Path
from typing import Any, TypeVar, Union

import yaml

T = TypeVar("T", bound="ConfigLoadable")


class ConfigLoadable:
    """Mixin class to mark a dataclass as loadable from external configs.
    
    Any dataclass inheriting from this class can be loaded from YAML, W&B,
    and other external sources using the provided class methods.
    
    The dataclass should use @dataclass(kw_only=True) decorator for best
    compatibility with partial config loading.
    """

    @classmethod
    def load_from_yaml(cls: type[T], filepath: Union[str, Path]) -> T:
        """Load configuration from a YAML file.

        Args:
            filepath: Path to the YAML configuration file.

        Returns:
            Instance of the dataclass with values populated from YAML.

        Raises:
            FileNotFoundError: If the YAML file does not exist.
            ValueError: If the YAML content cannot be converted to the dataclass type.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, "r") as f:
            data = yaml.safe_load(f)

        if data is None:
            data = {}

        return cls.load_from_dict(data)

    @classmethod
    def load_from_dict(cls: type[T], data: dict[str, Any]) -> T:
        """Load configuration from a dictionary.

        Args:
            data: Dictionary containing configuration values.

        Returns:
            Instance of the dataclass with values populated from the dictionary.

        Raises:
            ValueError: If required fields are missing or types don't match.
        """
        # Get all fields from the dataclass
        fields = {f.name: f for f in dataclasses.fields(cls)}  # type: ignore[arg-type]

        # Separate provided values from defaults
        kwargs = {}
        for field_name, field in fields.items():
            if field_name in data:
                value = data[field_name]
                # Recursively load nested ConfigLoadable objects
                if dataclasses.is_dataclass(field.type) and hasattr(
                    field.type, "load_from_dict"
                ):
                    kwargs[field_name] = field.type.load_from_dict(value)  # type: ignore[attr-defined]
                else:
                    kwargs[field_name] = value

        try:
            return cls(**kwargs)
        except TypeError as e:
            raise ValueError(
                f"Failed to instantiate {cls.__name__} with provided values: {e}"
            ) from e

    @classmethod
    def load_from_wandb(
        cls: type[T],
        wandb_entity: str,
        wandb_project: str,
        config_path: str = "config.yaml",
        run_name: str | None = None,
        artifact_type: str = "config",
    ) -> T:
        """Load configuration from Weights & Biases.

        Downloads a configuration artifact from W&B and loads it into the dataclass.

        Args:
            wandb_entity: Weights & Biases entity/username.
            wandb_project: Weights & Biases project name.
            config_path: Path within the artifact to the config file (default: "config.yaml").
            run_name: Specific run name to download from. If None, uses the latest run.
            artifact_type: Type of artifact to download (default: "config").

        Returns:
            Instance of the dataclass with values from W&B config.

        Raises:
            ImportError: If wandb is not installed.
            FileNotFoundError: If the artifact or config file is not found.
            ValueError: If the config cannot be loaded.
        """
        try:
            import wandb
        except ImportError as e:
            raise ImportError("wandb is required to load configs from W&B") from e

        api = wandb.Api()

        # Find the artifact
        if run_name:
            # Download from a specific run
            run = api.run(f"{wandb_entity}/{wandb_project}/{run_name}")
            artifacts = list(run.logged_artifacts())
        else:
            # Get latest artifact from the project
            artifacts = []
            for run in api.runs(f"{wandb_entity}/{wandb_project}"):
                artifacts.extend(run.logged_artifacts())

        # Filter artifacts by type and find the latest one
        config_artifacts = [
            a for a in artifacts if a.type == artifact_type
        ]
        if not config_artifacts:
            raise FileNotFoundError(
                f"No artifacts of type '{artifact_type}' found in "
                f"{wandb_entity}/{wandb_project}"
            )

        # Use the most recent one
        artifact = sorted(config_artifacts, key=lambda a: a.created_at)[-1]

        # Download and load the config
        artifact_dir = artifact.download()
        config_file = Path(artifact_dir) / config_path

        if not config_file.exists():
            raise FileNotFoundError(
                f"Config file '{config_path}' not found in artifact"
            )

        return cls.load_from_yaml(config_file)

    @classmethod
    def load_from_json(cls: type[T], filepath: Union[str, Path]) -> T:
        """Load configuration from a JSON file.

        Args:
            filepath: Path to the JSON configuration file.

        Returns:
            Instance of the dataclass with values populated from JSON.

        Raises:
            FileNotFoundError: If the JSON file does not exist.
            ValueError: If the JSON content cannot be converted to the dataclass type.
        """
        filepath = Path(filepath)
        if not filepath.exists():
            raise FileNotFoundError(f"Config file not found: {filepath}")

        with open(filepath, "r") as f:
            data = json.load(f)

        return cls.load_from_dict(data)

    def save_to_yaml(self, filepath: Union[str, Path], sort_keys: bool = False) -> None:
        """Save configuration to a YAML file.

        Args:
            filepath: Path where to save the YAML configuration.
            sort_keys: Whether to sort keys in the output.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict()
        with open(filepath, "w") as f:
            yaml.dump(data, f, sort_keys=sort_keys, default_flow_style=False)

    def save_to_json(self, filepath: Union[str, Path], indent: int = 2) -> None:
        """Save configuration to a JSON file.

        Args:
            filepath: Path where to save the JSON configuration.
            indent: Number of spaces for indentation.
        """
        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        data = self._to_dict()
        with open(filepath, "w") as f:
            json.dump(data, f, indent=indent)

    def _to_dict(self) -> dict[str, Any]:
        """Convert dataclass instance to dictionary, handling nested dataclasses."""
        result = {}
        for field in dataclasses.fields(self):  # type: ignore[arg-type]
            value = getattr(self, field.name)
            if dataclasses.is_dataclass(value) and hasattr(value, "_to_dict"):
                result[field.name] = value._to_dict()  # type: ignore[attr-defined]
            else:
                result[field.name] = value
        return result

    def override_from_config(self, config_path: Union[str, Path]) -> None:
        """Override this config's fields from a YAML/JSON file.
        
        Only fields present in the config file override this config's values.
        Missing fields retain their current values.
        
        Args:
            config_path: Path to YAML or JSON config file
        """
        config_path = Path(config_path)
        if config_path.suffix in ['.yaml', '.yml']:
            loaded_cfg = self.load_from_yaml(config_path)
        elif config_path.suffix == '.json':
            loaded_cfg = self.load_from_json(config_path)
        else:
            raise ValueError(f"Unsupported config file format: {config_path.suffix}")
        
        # Override fields that were loaded from the config file
        config_data = loaded_cfg._to_dict()
        for key, value in config_data.items():
            if hasattr(self, key):
                setattr(self, key, value)


def apply_config_overrides(
    cfg: T,
    overrides: dict[str, Any],
    recursive: bool = True,
) -> T:
    """Apply configuration overrides to a dataclass instance.

    Args:
        cfg: The configuration object to override.
        overrides: Dictionary of field names to new values.
        recursive: If True, recursively apply overrides to nested dataclasses.

    Returns:
        Modified configuration object (same instance).

    Raises:
        AttributeError: If attempting to set a non-existent field.
    """
    for key, value in overrides.items():
        if not hasattr(cfg, key):
            raise AttributeError(
                f"Configuration {type(cfg).__name__} has no field '{key}'"
            )

        current_value = getattr(cfg, key)
        if (
            recursive
            and dataclasses.is_dataclass(current_value)
            and isinstance(value, dict)
        ):
            # Recursively apply overrides to nested dataclass
            apply_config_overrides(current_value, value, recursive=True)  # type: ignore[arg-type]
        else:
            setattr(cfg, key, value)

    return cfg


def merge_configs(
    base_cfg: T,
    override_cfg: T,
    recursive: bool = True,
) -> T:
    """Merge two configuration objects.

    Args:
        base_cfg: The base configuration.
        override_cfg: Configuration with values to override.
        recursive: If True, recursively merge nested dataclasses.

    Returns:
        Modified base configuration (same instance).
    """
    overrides = {}
    for field in dataclasses.fields(override_cfg):  # type: ignore[arg-type]
        override_value = getattr(override_cfg, field.name)
        base_value = getattr(base_cfg, field.name)

        if (
            recursive
            and dataclasses.is_dataclass(override_value)
            and dataclasses.is_dataclass(base_value)
        ):
            # Recursively merge nested dataclasses
            merge_configs(base_value, override_value, recursive=True)  # type: ignore[arg-type]
        elif override_value != field.default and override_value != field.default_factory:
            overrides[field.name] = override_value

    return apply_config_overrides(base_cfg, overrides, recursive=False)
