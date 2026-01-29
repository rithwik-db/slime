"""Configuration validation logic."""

import logging
from pathlib import Path
from typing import List

from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)


class ConfigValidationError(Exception):
    """Raised when configuration validation fails."""
    pass


def validate_config(config: DictConfig) -> DictConfig:
    """
    Validate configuration and apply derived defaults.

    This mirrors logic from slime_validate_args() in arguments.py.

    Args:
        config: OmegaConf DictConfig to validate

    Returns:
        Validated config with derived defaults applied

    Raises:
        ConfigValidationError: If validation fails
    """
    errors: List[str] = []

    # Ensure config is resolved
    OmegaConf.resolve(config)

    # --- Checkpoint validation ---
    if config.checkpoint.hf_checkpoint:
        hf_path = Path(config.checkpoint.hf_checkpoint)
        if not hf_path.exists():
            errors.append(f"hf_checkpoint path does not exist: {hf_path}")

    # --- KL coefficient validation ---
    if config.algorithm.kl_coef != 0 or config.algorithm.use_kl_loss:
        if config.checkpoint.ref_load:
            ref_path = Path(config.checkpoint.ref_load)
            if not ref_path.exists():
                errors.append(
                    f"ref_load is required when using KL loss/coef but path does not exist: {ref_path}"
                )
        else:
            errors.append("ref_load is required when kl_coef != 0 or use_kl_loss is True")

    # --- Dynamic batch size validation ---
    if config.data.use_dynamic_batch_size:
        if config.data.max_tokens_per_gpu is None:
            errors.append("max_tokens_per_gpu must be set when use_dynamic_batch_size is True")

    # --- Advantage estimator validation ---
    valid_estimators = ["gae", "grpo", "reinforce", "rloo"]
    if config.algorithm.advantage_estimator not in valid_estimators:
        errors.append(
            f"Invalid advantage_estimator: {config.algorithm.advantage_estimator}. "
            f"Must be one of: {valid_estimators}"
        )

    # --- Apply derived defaults ---

    # eps_clip_high defaults to eps_clip if not set
    if config.algorithm.eps_clip_high is None:
        config.algorithm.eps_clip_high = config.algorithm.eps_clip

    # critic_num_nodes defaults to actor_num_nodes if not set
    if config.cluster.critic_num_nodes is None:
        config.cluster.critic_num_nodes = config.cluster.actor_num_nodes

    # critic_num_gpus_per_node defaults to actor_num_gpus_per_node if not set
    if config.cluster.critic_num_gpus_per_node is None:
        config.cluster.critic_num_gpus_per_node = config.cluster.actor_num_gpus_per_node

    # --- Training backend validation ---
    valid_backends = ["megatron", "fsdp"]
    if config.train.backend not in valid_backends:
        errors.append(
            f"Invalid train_backend: {config.train.backend}. "
            f"Must be one of: {valid_backends}"
        )

    # --- Launch method validation ---
    valid_launch_methods = ["ray_job_submit", "torchrun"]
    if config.launcher.method not in valid_launch_methods:
        errors.append(
            f"Invalid launch_method: {config.launcher.method}. "
            f"Must be one of: {valid_launch_methods}"
        )

    # Raise all errors at once
    if errors:
        error_msg = "Configuration validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        raise ConfigValidationError(error_msg)

    logger.info("Configuration validation passed")
    return config
