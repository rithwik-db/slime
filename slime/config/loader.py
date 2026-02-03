"""
Configuration loader for Slime training.

Handles YAML loading, CLI integration, and conversion to argparse namespace.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import yaml
from pydantic import ValidationError

from slime.config.models import (
    SlimeConfig,
    ClusterConfig,
    TrainConfig,
    RolloutConfig,
    FaultToleranceConfig,
    DataConfig,
    EvaluationConfig,
    AlgorithmConfig,
    RouterConfig,
    WandbConfig,
    TensorboardConfig,
    DebugConfig,
    NetworkConfig,
    RewardConfig,
    RolloutBufferConfig,
    MTPConfig,
    PrefillDecodeConfig,
    CIConfig,
    SGLangConfig,
)

logger = logging.getLogger(__name__)


# =============================================================================
# Auto-generated mapping from flat arg names to (section, field_name)
# =============================================================================

# Section name -> config class mapping (matches SlimeConfig field names)
_SECTION_CONFIGS: list[tuple[str, type]] = [
    ("cluster", ClusterConfig),
    ("train", TrainConfig),
    ("rollout", RolloutConfig),
    ("fault_tolerance", FaultToleranceConfig),
    ("data", DataConfig),
    ("evaluation", EvaluationConfig),
    ("algorithm", AlgorithmConfig),
    ("router", RouterConfig),
    ("wandb", WandbConfig),
    ("tensorboard", TensorboardConfig),
    ("debug", DebugConfig),
    ("network", NetworkConfig),
    ("reward", RewardConfig),
    ("rollout_buffer", RolloutBufferConfig),
    ("mtp", MTPConfig),
    ("prefill_decode", PrefillDecodeConfig),
    ("ci", CIConfig),
    ("sglang", SGLangConfig),
]


def _build_arg_to_section_mapping() -> dict[str, tuple[str, str]]:
    """
    Auto-generate ARG_TO_SECTION mapping from Pydantic model fields.

    Iterates over each section config and maps field names to (section, field).
    """
    mapping: dict[str, tuple[str, str]] = {}

    for section_name, config_class in _SECTION_CONFIGS:
        for field_name in config_class.model_fields:
            mapping[field_name] = (section_name, field_name)

    return mapping


ARG_TO_SECTION: dict[str, tuple[str, str]] = _build_arg_to_section_mapping()


def load_yaml_config(yaml_path: str | Path) -> dict[str, Any]:
    """Load and parse a YAML configuration file."""
    path = Path(yaml_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {yaml_path}")

    with open(path) as f:
        data = yaml.safe_load(f) or {}

    if not isinstance(data, dict):
        raise ValueError(f"Config file must contain a YAML mapping, got {type(data).__name__}")

    return data


def deep_merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """
    Deep merge two dictionaries, with override taking precedence.

    For nested dicts, recursively merges. For other types, override replaces base.
    """
    result = base.copy()
    for key, value in override.items():
        if key in result and isinstance(result[key], dict) and isinstance(value, dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def flat_dict_to_nested(flat: dict[str, Any]) -> dict[str, Any]:
    """
    Convert a flat dictionary of CLI args to nested config structure.

    Uses ARG_TO_SECTION mapping to determine which section each arg belongs to.
    Unknown args are placed at the root level for passthrough.
    """
    result: dict[str, Any] = {}

    for key, value in flat.items():
        if value is None:
            # Skip None values to allow YAML defaults
            continue
        if value == "":
            # Skip empty string defaults to allow YAML/Pydantic defaults
            continue

        if key in ARG_TO_SECTION:
            section, field = ARG_TO_SECTION[key]
            if section not in result:
                result[section] = {}
            result[section][field] = value
        else:
            # Put unmapped args at root level (passthrough for unmapped args)
            result[key] = value

    return result


def load_config(
    yaml_path: str | Path | None = None,
    cli_overrides: dict[str, Any] | None = None,
) -> SlimeConfig:
    """
    Load configuration from YAML, then apply CLI overrides.

    Priority (highest to lowest):
    1. CLI arguments (cli_overrides)
    2. YAML file
    3. Pydantic defaults

    Args:
        yaml_path: Path to YAML configuration file (optional)
        cli_overrides: Flat dictionary of CLI argument overrides (optional)

    Returns:
        Validated SlimeConfig object
    """
    config_dict: dict[str, Any] = {}

    # Load YAML if provided
    if yaml_path:
        yaml_dict = load_yaml_config(yaml_path)
        config_dict = deep_merge(config_dict, yaml_dict)
        logger.info(f"Loaded config from YAML: {yaml_path}")

    # Apply CLI overrides if provided
    if cli_overrides:
        # Convert flat CLI args to nested structure
        nested_overrides = flat_dict_to_nested(cli_overrides)
        config_dict = deep_merge(config_dict, nested_overrides)

    # Validate and create config
    try:
        config = SlimeConfig(**config_dict)
    except ValidationError as e:
        logger.error(f"Configuration validation failed:\n{e}")
        raise

    return config


def config_to_namespace(config: SlimeConfig) -> argparse.Namespace:
    """
    Convert a SlimeConfig to an argparse.Namespace for backward compatibility.

    This flattens the nested config structure into a flat namespace that
    existing code expects from parse_args().
    """
    flat_dict = config.to_flat_dict()
    return argparse.Namespace(**flat_dict)


def namespace_to_cli_overrides(args: argparse.Namespace) -> dict[str, Any]:
    """
    Convert an argparse.Namespace to a flat dictionary suitable for CLI overrides.

    Filters out None values and converts to the flat format expected by load_config.
    """
    return {k: v for k, v in vars(args).items() if v is not None}


def create_train_yaml_parser() -> argparse.ArgumentParser:
    """Create a minimal parser that captures --train-yaml before full parsing."""
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument(
        "--train-yaml",
        type=str,
        default=None,
        help="Path to YAML configuration file. When provided, config is loaded from YAML and CLI args override.",
    )
    return parser
