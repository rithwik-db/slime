"""Slime configuration module using OmegaConf."""

from .loader import load_config, load_yaml_as_dict, save_config
from .base import SlimeConfig
from .validation import validate_config, ConfigValidationError
from .converter import config_to_namespace, namespace_to_config, apply_yaml_defaults_to_namespace

__all__ = [
    "load_config",
    "load_yaml_as_dict",
    "save_config",
    "SlimeConfig",
    "validate_config",
    "ConfigValidationError",
    "config_to_namespace",
    "namespace_to_config",
    "apply_yaml_defaults_to_namespace",
]
