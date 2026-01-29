"""Configuration loading utilities using OmegaConf."""

import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Union

from omegaconf import OmegaConf, DictConfig

from .base import SlimeConfig

logger = logging.getLogger(__name__)


def load_config(
    config_path: Optional[str] = None,
    cli_overrides: Optional[List[str]] = None,
    resolve: bool = True,
) -> DictConfig:
    """
    Load configuration from YAML file with OmegaConf.

    Supports variable interpolation like ${pretrain_model_name} and ${max_model_len}.

    Args:
        config_path: Path to YAML config file
        cli_overrides: List of CLI overrides in dot notation (e.g., ["optimizer.lr=1e-5"])
        resolve: Whether to resolve interpolations immediately

    Returns:
        OmegaConf DictConfig with (optionally) resolved values

    Example:
        >>> config = load_config("configs/training/qwen3-4b.yaml")
        >>> print(config.optimizer.lr)
        1e-6
    """
    # Start with structured defaults
    base_config = OmegaConf.structured(SlimeConfig)

    if config_path:
        config_path = Path(config_path)
        if not config_path.exists():
            raise FileNotFoundError(f"Config file not found: {config_path}")

        file_config = OmegaConf.load(config_path)

        # Handle nested 'parameters:' key from target YAML format
        if "parameters" in file_config:
            file_config = file_config.parameters

        # Merge file config into base
        base_config = OmegaConf.merge(base_config, file_config)
        logger.info(f"Loaded config from {config_path}")

    # Apply CLI overrides
    if cli_overrides:
        cli_config = OmegaConf.from_dotlist(cli_overrides)
        base_config = OmegaConf.merge(base_config, cli_config)
        logger.debug(f"Applied CLI overrides: {cli_overrides}")

    # Resolve interpolations if requested
    if resolve:
        OmegaConf.resolve(base_config)

    return base_config


def load_yaml_as_dict(
    config_path: str,
    resolve: bool = True,
) -> Dict[str, Any]:
    """
    Load YAML config file and return as a flat dictionary.

    This is useful for integrating with existing argparse-based code
    by setting parser defaults from YAML values.

    Args:
        config_path: Path to YAML config file
        resolve: Whether to resolve interpolations

    Returns:
        Flattened dictionary suitable for argparse defaults
    """
    config = load_config(config_path, resolve=resolve)
    return _flatten_config(OmegaConf.to_container(config, resolve=resolve))


def _flatten_config(
    config: Dict[str, Any],
    parent_key: str = "",
    sep: str = "_",
    _top_level_keys: Optional[set] = None,
) -> Dict[str, Any]:
    """
    Flatten nested config dictionary for argparse compatibility.

    Converts nested keys to flat keys:
        {'optimizer': {'lr': 1e-6}} -> {'optimizer_lr': 1e-6}

    Also creates shorthand keys without prefix for top-level access:
        {'optimizer': {'lr': 1e-6}} -> {'lr': 1e-6, 'optimizer_lr': 1e-6}

    Top-level keys take precedence over shorthand keys from nested dicts.
    """
    # Track top-level keys to avoid overwriting them with shorthand keys
    if _top_level_keys is None:
        _top_level_keys = {k.replace("-", "_") for k in config.keys() if not isinstance(config[k], dict)}

    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            # Recursively flatten nested dicts
            items.extend(_flatten_config(value, new_key, sep=sep, _top_level_keys=_top_level_keys).items())
            # Also add shorthand keys for leaf values (but don't overwrite top-level keys)
            for nested_key, nested_value in value.items():
                if not isinstance(nested_value, dict):
                    shorthand_key = nested_key.replace("-", "_")
                    # Only add shorthand if it won't overwrite a top-level key
                    if shorthand_key not in _top_level_keys:
                        items.append((shorthand_key, nested_value))
        else:
            items.append((new_key.replace("-", "_"), value))

    return dict(items)


def save_config(config: Union[DictConfig, SlimeConfig], path: str) -> None:
    """Save configuration to YAML file."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(config, SlimeConfig):
        config = OmegaConf.structured(config)

    OmegaConf.save(config, path)
    logger.info(f"Saved config to {path}")
