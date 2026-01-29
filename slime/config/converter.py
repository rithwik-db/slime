"""Convert between OmegaConf config and argparse Namespace."""

import argparse
import logging
from typing import Any, Dict, Optional

from omegaconf import OmegaConf, DictConfig

from .base import SlimeConfig

logger = logging.getLogger(__name__)


def config_to_namespace(config: DictConfig) -> argparse.Namespace:
    """
    Convert OmegaConf config to argparse Namespace.

    This allows using YAML configs with existing code that expects
    argparse Namespace objects.
    """
    flat_dict = _deep_flatten(OmegaConf.to_container(config, resolve=True))
    return argparse.Namespace(**flat_dict)


def namespace_to_config(args: argparse.Namespace) -> DictConfig:
    """
    Convert argparse Namespace to OmegaConf config.

    Useful for saving current args as YAML for reproducibility.
    """
    return OmegaConf.create(vars(args))


def apply_yaml_defaults_to_namespace(
    args: argparse.Namespace,
    yaml_config: Dict[str, Any],
    explicit_args: Optional[set] = None,
) -> argparse.Namespace:
    """
    Apply YAML config values to parsed args, respecting CLI overrides.

    CLI arguments explicitly provided take precedence over YAML values.

    Args:
        args: Parsed argparse Namespace
        yaml_config: Flattened YAML config dictionary
        explicit_args: Set of argument names explicitly provided on CLI

    Returns:
        Updated Namespace with YAML defaults applied
    """
    explicit_args = explicit_args or set()

    for key, value in yaml_config.items():
        # Convert YAML key (may have dashes) to argparse dest (underscores)
        argparse_key = key.replace("-", "_")

        # Skip if explicitly set via CLI
        if argparse_key in explicit_args:
            continue

        # Only apply if attribute exists and current value is None or default
        if hasattr(args, argparse_key):
            current_value = getattr(args, argparse_key)
            if current_value is None:
                setattr(args, argparse_key, value)
                logger.debug(f"Set {argparse_key} = {value} from YAML")

    return args


def _deep_flatten(d: Dict[str, Any], parent_key: str = "", sep: str = "_") -> Dict[str, Any]:
    """Deeply flatten nested dictionary."""
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_deep_flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


# Mapping from YAML config keys to argparse argument names
# This handles cases where naming conventions differ
YAML_TO_ARGPARSE_MAP = {
    # Cluster
    "cluster_actor_num_nodes": "actor_num_nodes",
    "cluster_actor_num_gpus_per_node": "actor_num_gpus_per_node",
    "cluster_rollout_num_gpus": "rollout_num_gpus",
    "cluster_rollout_num_gpus_per_engine": "rollout_num_gpus_per_engine",
    "cluster_colocate": "colocate",

    # Checkpoint
    "checkpoint_hf_checkpoint": "hf_checkpoint",
    "checkpoint_load": "load",
    "checkpoint_save": "save",
    "checkpoint_ref_load": "ref_load",
    "checkpoint_save_interval": "save_interval",

    # Rollout
    "rollout_prompt_data": "prompt_data",
    "rollout_input_key": "input_key",
    "rollout_label_key": "label_key",
    "rollout_temperature": "rollout_temperature",
    "rollout_batch_size": "rollout_batch_size",
    "rollout_n_samples_per_prompt": "n_samples_per_prompt",

    # Algorithm
    "algorithm_advantage_estimator": "advantage_estimator",
    "algorithm_eps_clip": "eps_clip",
    "algorithm_kl_coef": "kl_coef",
    "algorithm_use_kl_loss": "use_kl_loss",

    # Optimizer
    "optimizer_lr": "lr",
    "optimizer_name": "optimizer",
    "optimizer_weight_decay": "weight_decay",

    # Logging
    "logging_wandb_enabled": "use_wandb",
    "logging_tensorboard_enabled": "use_tensorboard",
    "logging_mlflow_enabled": "use_mlflow",

    # Launcher
    "launcher_method": "launch_method",
    "launcher_spmd_ray_timeout": "spmd_ray_timeout",

    # Train
    "train_backend": "train_backend",
    "train_num_rollout": "num_rollout",
}


def map_yaml_key_to_argparse(yaml_key: str) -> str:
    """Map YAML config key to corresponding argparse argument name."""
    return YAML_TO_ARGPARSE_MAP.get(yaml_key, yaml_key)
