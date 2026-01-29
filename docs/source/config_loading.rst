Configuration Loading
=====================

Overview
--------

The unified config loading system provides a flexible, standardized way to parse and manage configurations from external sources like YAML files, JSON files, and Weights & Biases (W&B). It uses a mixin-based approach with the ``ConfigLoadable`` class to mark dataclasses as loadable from external sources.

Key Features
------------

- **YAML Support**: Load and save configurations from/to YAML files
- **JSON Support**: Load and save configurations from/to JSON files  
- **W&B Integration**: Load configurations directly from Weights & Biases artifacts
- **Nested Dataclasses**: Automatically handle nested dataclass configurations
- **Config Overrides**: Apply partial overrides to existing configurations
- **Config Merging**: Merge two configurations with intelligent override handling
- **Type Checking**: Validates that loaded values match expected types

Quick Start
-----------

Mark Your Dataclass
~~~~~~~~~~~~~~~~~~~~

Inherit from ``ConfigLoadable`` to enable external config loading:

.. code-block:: python

    from dataclasses import dataclass
    from mjlab.utils import ConfigLoadable

    @dataclass(kw_only=True)
    class MyConfig(ConfigLoadable):
        learning_rate: float = 0.001
        num_epochs: int = 100
        batch_size: int = 32

Load from YAML
~~~~~~~~~~~~~~

.. code-block:: python

    # Load from file
    cfg = MyConfig.load_from_yaml("config.yaml")

    # Load from dictionary
    cfg = MyConfig.load_from_dict({
        "learning_rate": 0.0005,
        "num_epochs": 200
    })

Save Configuration
~~~~~~~~~~~~~~~~~~

.. code-block:: python

    # Save to YAML
    cfg.save_to_yaml("output_config.yaml")

    # Save to JSON
    cfg.save_to_json("output_config.json")

Detailed Usage
--------------

Loading from YAML
~~~~~~~~~~~~~~~~~

YAML files support hierarchical configuration:

.. code-block:: yaml

    # config.yaml
    learning_rate: 0.0001
    num_epochs: 500
    batch_size: 128

Load it:

.. code-block:: python

    cfg = MyConfig.load_from_yaml("config.yaml")

Fields that aren't specified in the YAML will use their default values from the dataclass.

Nested Dataclasses
~~~~~~~~~~~~~~~~~~

For complex configurations with nested dataclasses:

.. code-block:: python

    from dataclasses import dataclass, field

    @dataclass(kw_only=True)
    class ModelConfig(ConfigLoadable):
        hidden_dim: int = 256
        num_layers: int = 3

    @dataclass(kw_only=True)
    class TrainingConfig(ConfigLoadable):
        model: ModelConfig = field(default_factory=ModelConfig)
        learning_rate: float = 0.001

YAML structure:

.. code-block:: yaml

    learning_rate: 0.0005
    model:
      hidden_dim: 512
      num_layers: 5

Load:

.. code-block:: python

    cfg = TrainingConfig.load_from_yaml("config.yaml")
    print(cfg.model.hidden_dim)  # 512

Loading from W&B
~~~~~~~~~~~~~~~~

Download configurations from Weights & Biases:

.. code-block:: python

    cfg = MyConfig.load_from_wandb(
        wandb_entity="your_entity",
        wandb_project="your_project",
        config_path="config.yaml",
    )

Requirements:

- W&B artifact must be uploaded as type ``"config"``
- The artifact must contain the specified config file (default: ``config.yaml``)

Applying Configuration Overrides
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

Apply selective overrides to an existing configuration:

.. code-block:: python

    from mjlab.utils import apply_config_overrides

    cfg = MyConfig()
    overrides = {
        "learning_rate": 0.0001,
        "num_epochs": 1000
    }
    apply_config_overrides(cfg, overrides)

For nested configs:

.. code-block:: python

    overrides = {
        "model": {
            "hidden_dim": 768,
            "num_layers": 12
        }
    }
    apply_config_overrides(cfg, overrides, recursive=True)

Merging Configurations
~~~~~~~~~~~~~~~~~~~~~~

Intelligently merge two configurations:

.. code-block:: python

    from mjlab.utils import merge_configs

    base_cfg = TrainingConfig()
    override_cfg = TrainingConfig(
        model=ModelConfig(hidden_dim=512),
        learning_rate=0.0001
    )

    result = merge_configs(base_cfg, override_cfg, recursive=True)

Best Practices
--------------

Use kw_only for Dataclasses
~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @dataclass(kw_only=True)  # Important!
    class MyConfig(ConfigLoadable):
        param1: int = 10
        param2: str = "default"

This ensures:

- Partial loading works correctly
- Field order doesn't matter
- Clear separation between required and optional fields

Provide Sensible Defaults
~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @dataclass(kw_only=True)
    class MyConfig(ConfigLoadable):
        # Good: clear defaults that work for common cases
        learning_rate: float = 0.001
        num_epochs: int = 100
        device: str = "cuda:0"

Document Configuration Options
~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

.. code-block:: python

    @dataclass(kw_only=True)
    class MyConfig(ConfigLoadable):
        """Configuration for training.
        
        Attributes:
            learning_rate: Initial learning rate (default: 0.001)
            num_epochs: Number of training epochs (default: 100)
            device: Device to train on, 'cuda:0' or 'cpu' (default: 'cuda:0')
        """
        learning_rate: float = 0.001
        num_epochs: int = 100
        device: str = "cuda:0"

Multiple Config Files
~~~~~~~~~~~~~~~~~~~~~

You have several options for organizing multiple configs:

**Option 1: Separate YAML files** (Recommended)

.. code-block:: python

    motion_cmd = MotionCommandCfg.load_from_yaml("conf/motion.yaml")
    env_cfg = ManagerBasedRlEnvCfg.load_from_yaml("conf/env.yaml")
    runner_cfg = RslRlOnPolicyRunnerCfg.load_from_yaml("conf/runner.yaml")

**Option 2: Single YAML with sections**

.. code-block:: yaml

    # conf/full_config.yaml
    motion:
      motion_file: artifacts/walk1:v0/motion.npz
      has_state_estimation: true
      
    env:
      episode_length_s: 20.0
      decimation: 4
      
    runner:
      max_iterations: 5000
      learning_rate: 0.001

Then load specific sections:

.. code-block:: python

    import yaml

    with open("conf/full_config.yaml") as f:
        data = yaml.safe_load(f)

    motion_cmd = MotionCommandCfg.load_from_dict(data["motion"])
    env_overrides = data["env"]
    runner_overrides = data["runner"]

API Reference
-------------

ConfigLoadable Methods
~~~~~~~~~~~~~~~~~~~~~~

.. py:class:: ConfigLoadable

   Mixin class to mark a dataclass as loadable from external configs.

   .. py:method:: load_from_yaml(filepath: str | Path) -> T
      :classmethod:

      Load configuration from a YAML file.

      :param filepath: Path to the YAML configuration file
      :returns: Instance of the dataclass with values populated from YAML
      :raises FileNotFoundError: If the YAML file does not exist
      :raises ValueError: If the YAML content cannot be converted to the dataclass type

   .. py:method:: load_from_json(filepath: str | Path) -> T
      :classmethod:

      Load configuration from a JSON file.

      :param filepath: Path to the JSON configuration file
      :returns: Instance of the dataclass with values populated from JSON

   .. py:method:: load_from_dict(data: dict[str, Any]) -> T
      :classmethod:

      Load configuration from a dictionary.

      :param data: Dictionary containing configuration values
      :returns: Instance of the dataclass with values populated from the dictionary

   .. py:method:: load_from_wandb(wandb_entity: str, wandb_project: str, config_path: str = "config.yaml", run_name: str | None = None, artifact_type: str = "config") -> T
      :classmethod:

      Load configuration from Weights & Biases.

      :param wandb_entity: Weights & Biases entity/username
      :param wandb_project: Weights & Biases project name
      :param config_path: Path within the artifact to the config file
      :param run_name: Specific run name to download from (if None, uses latest)
      :param artifact_type: Type of artifact to download
      :returns: Instance of the dataclass with values from W&B config

   .. py:method:: save_to_yaml(filepath: str | Path, sort_keys: bool = False) -> None

      Save configuration to a YAML file.

      :param filepath: Path where to save the YAML configuration
      :param sort_keys: Whether to sort keys in the output

   .. py:method:: save_to_json(filepath: str | Path, indent: int = 2) -> None

      Save configuration to a JSON file.

      :param filepath: Path where to save the JSON configuration
      :param indent: Number of spaces for indentation

   .. py:method:: override_from_config(config_path: str | Path) -> None

      Override this config's fields from a YAML/JSON file.
      
      Only fields present in the config file override this config's values.
      Missing fields retain their current values.

      :param config_path: Path to YAML or JSON config file

Helper Functions
~~~~~~~~~~~~~~~~

.. py:function:: apply_config_overrides(cfg: T, overrides: dict[str, Any], recursive: bool = True) -> T

   Apply configuration overrides to a dataclass instance.

   :param cfg: The configuration object to override
   :param overrides: Dictionary of field names to new values
   :param recursive: If True, recursively apply overrides to nested dataclasses
   :returns: Modified configuration object (same instance)
   :raises AttributeError: If attempting to set a non-existent field

.. py:function:: merge_configs(base_cfg: T, override_cfg: T, recursive: bool = True) -> T

   Merge two configuration objects.

   :param base_cfg: The base configuration
   :param override_cfg: Configuration with values to override
   :param recursive: If True, recursively merge nested dataclasses
   :returns: Modified base configuration (same instance)

Examples
--------

See the ``scripts/examples/`` directory for complete working examples:

- ``g1_config_example.py`` - G1 tracking environment configuration
- ``config_loading_example.py`` - Basic config loading examples
- ``config_integration_example.py`` - Integration with mjlab configs
