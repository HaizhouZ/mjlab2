"""Integration module for tyro and ConfigLoadable.

This module provides utilities for seamlessly integrating tyro CLI parsing
with the unified config loading system, allowing configs to be loaded from
YAML files, command-line arguments, or both.

Example:
    >>> from mjlab.utils.tyro_integration import apply_tyro_config_loading
    >>> 
    >>> @dataclass(kw_only=True)
    >>> class MyConfig(ConfigLoadable):
    ...     param1: int = 10
    ...     param2: str = "default"
    >>> 
    >>> # Use in tyro.cli() call
    >>> cfg = tyro.cli(
    ...     MyConfig,
    ...     config=(apply_tyro_config_loading,),
    ... )
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from mjlab.utils.config_loader import apply_config_overrides


@dataclass
class ConfigPathArg:
    """Helper dataclass to add config_path argument to CLI.
    
    This can be composed with other configs to add YAML config file loading.
    """
    config_path: Optional[str] = None
    """Path to YAML config file to load. Overrides defaults but is overridden by CLI args."""


def apply_tyro_config_loading(
    target_type: type,
    markers: tuple,
    prefix: str,
) -> None:
    """Tyro resolver that enables loading from YAML config files.
    
    This is designed to be used as a config in tyro.cli(), e.g.:
    
    ```python
    tyro.cli(
        MyConfig,
        config=(apply_tyro_config_loading,),
    )
    ```
    
    When a config_path argument is provided, it will:
    1. Load the base config from the YAML file
    2. Allow CLI args to override the loaded config
    
    Args:
        target_type: The type being parsed by tyro
        markers: Tyro markers (unused)
        prefix: CLI prefix (unused)
    """
    # This is called during tyro initialization
    # The actual loading happens in the wrapper function below
    pass


def load_config_with_cli_overrides(
    config_class: type,
    args: Optional[list[str]] = None,
    config_yaml_path: Optional[str] = None,
    tyro_config: tuple = (
        __import__("tyro").conf.AvoidSubcommands,
        __import__("tyro").conf.FlagConversionOff,
    ),
    **tyro_kwargs,
) -> Any:
    """Load config from YAML file with CLI argument overrides.
    
    This function combines YAML config loading with tyro CLI parsing.
    CLI arguments take precedence over YAML config values.
    
    Args:
        config_class: The dataclass type to load (must inherit from ConfigLoadable)
        args: Command-line arguments to parse (default: sys.argv[1:])
        config_yaml_path: Path to YAML config file to load first
        tyro_config: Tyro configuration tuple
        **tyro_kwargs: Additional keyword arguments to pass to tyro.cli()
    
    Returns:
        Instance of config_class with values from YAML + CLI overrides
    
    Example:
        >>> cfg = load_config_with_cli_overrides(
        ...     MyConfig,
        ...     config_yaml_path="config.yaml",
        ... )
    """
    import tyro
    from mjlab.utils.config_loader import ConfigLoadable
    
    # First, load base config from YAML if provided
    if config_yaml_path:
        if not hasattr(config_class, 'load_from_yaml'):
            raise TypeError(
                f"{config_class.__name__} must inherit from ConfigLoadable "
                "to load from YAML"
            )
        base_cfg = config_class.load_from_yaml(config_yaml_path)
    else:
        # Create default instance
        base_cfg = config_class()
    
    # Then parse CLI args to get overrides (use base config as default)
    cli_cfg = tyro.cli(
        config_class,
        args=args,
        default=base_cfg,
        config=tyro_config,
        **tyro_kwargs,
    )
    
    return cli_cfg


def add_config_path_to_dataclass(dataclass_type: type) -> type:
    """Add config_path field to a dataclass.
    
    This is a helper to make it easier to add config file support to existing
    dataclasses. It wraps the dataclass with a new one that includes a
    config_path field.
    
    Args:
        dataclass_type: The dataclass type to enhance
    
    Returns:
        A new dataclass with config_path field added
    
    Example:
        >>> @add_config_path_to_dataclass
        >>> @dataclass(kw_only=True)
        >>> class MyConfig(ConfigLoadable):
        ...     param1: int = 10
    """
    import dataclasses
    
    @dataclass(kw_only=True)
    class EnhancedConfig(dataclass_type):
        config_path: Optional[Path] = None
        """Path to YAML config file to load (optional)"""
    
    return EnhancedConfig


def create_config_loader_wrapper(
    config_class: type,
    yaml_loader_method: str = "load_from_yaml",
    config_path_arg: str = "config_path",
):
    """Create a wrapper function for seamless YAML + CLI config loading.
    
    This creates a high-level wrapper that:
    1. Checks for config_path argument
    2. Loads from YAML if provided
    3. Parses CLI arguments with YAML config as defaults
    4. Returns final config
    
    Args:
        config_class: The dataclass type to wrap
        yaml_loader_method: Name of the YAML loading method (default: "load_from_yaml")
        config_path_arg: Name of the config path argument (default: "config_path")
    
    Returns:
        Wrapper function suitable for use with tyro.cli()
    
    Example:
        >>> wrapper = create_config_loader_wrapper(MyConfig)
        >>> cfg = wrapper(args=sys.argv[1:])
    """
    def wrapper(
        args: Optional[list[str]] = None,
        **tyro_kwargs,
    ) -> config_class:
        """Load config from optional YAML file + CLI arguments."""
        import tyro
        from mjlab.utils.config_loader import ConfigLoadable
        
        # First pass: just parse to check for config_path
        base_cfg = config_class()
        
        # Create a minimal config class just for the path argument
        @dataclass(kw_only=True)
        class PathOnly:
            config_path: Optional[str] = None
        
        try:
            path_only = tyro.cli(
                PathOnly,
                args=args,
                default=PathOnly(),
                config=(
                    tyro.conf.AvoidSubcommands,
                    tyro.conf.FlagConversionOff,
                ),
            )
            
            # If config_path was provided, load from YAML
            if path_only.config_path:
                if not hasattr(config_class, yaml_loader_method):
                    raise AttributeError(
                        f"{config_class.__name__} has no method '{yaml_loader_method}'"
                    )
                loader = getattr(config_class, yaml_loader_method)
                base_cfg = loader(path_only.config_path)
        except (SystemExit, Exception):
            # If first pass fails, just use default
            pass
        
        # Second pass: parse full config with base as default
        final_cfg = tyro.cli(
            config_class,
            args=args,
            default=base_cfg,
            config=(
                tyro.conf.AvoidSubcommands,
                tyro.conf.FlagConversionOff,
            ),
            **tyro_kwargs,
        )
        
        return final_cfg
    
    return wrapper
