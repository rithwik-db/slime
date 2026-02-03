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

from slime.config.models import SlimeConfig

logger = logging.getLogger(__name__)


# =============================================================================
# Mapping from flat arg names to (section, field_name)
# =============================================================================

# This maps argparse argument names to their config section and field
ARG_TO_SECTION: dict[str, tuple[str, str]] = {
    # Cluster
    "actor_num_nodes": ("cluster", "actor_num_nodes"),
    "actor_num_gpus_per_node": ("cluster", "actor_num_gpus_per_node"),
    "critic_num_nodes": ("cluster", "critic_num_nodes"),
    "critic_num_gpus_per_node": ("cluster", "critic_num_gpus_per_node"),
    "rollout_num_gpus": ("cluster", "rollout_num_gpus"),
    "rollout_num_gpus_per_engine": ("cluster", "rollout_num_gpus_per_engine"),
    "num_gpus_per_node": ("cluster", "num_gpus_per_node"),
    "colocate": ("cluster", "colocate"),
    "offload": ("cluster", "offload"),
    "offload_train": ("cluster", "offload_train"),
    "offload_rollout": ("cluster", "offload_rollout"),
    "distributed_backend": ("cluster", "distributed_backend"),
    "distributed_timeout_minutes": ("cluster", "distributed_timeout_minutes"),
    # Train
    "train_backend": ("train", "train_backend"),
    "qkv_format": ("train", "qkv_format"),
    "true_on_policy_mode": ("train", "true_on_policy_mode"),
    "train_env_vars": ("train", "train_env_vars"),
    "train_memory_margin_bytes": ("train", "train_memory_margin_bytes"),
    "enable_weights_backuper": ("train", "enable_weights_backuper"),
    "megatron_to_hf_mode": ("train", "megatron_to_hf_mode"),
    "custom_model_provider_path": ("train", "custom_model_provider_path"),
    "recompute_loss_function": ("train", "recompute_loss_function"),
    "log_probs_chunk_size": ("train", "log_probs_chunk_size"),
    "only_train_params_name_list": ("train", "only_train_params_name_list"),
    "freeze_params_name_list": ("train", "freeze_params_name_list"),
    # Rollout
    "hf_checkpoint": ("rollout", "hf_checkpoint"),
    "model_name": ("rollout", "model_name"),
    "rollout_function_path": ("rollout", "rollout_function_path"),
    "rollout_temperature": ("rollout", "rollout_temperature"),
    "rollout_top_p": ("rollout", "rollout_top_p"),
    "rollout_top_k": ("rollout", "rollout_top_k"),
    "rollout_max_context_len": ("rollout", "rollout_max_context_len"),
    "rollout_max_prompt_len": ("rollout", "rollout_max_prompt_len"),
    "rollout_max_response_len": ("rollout", "rollout_max_response_len"),
    "rollout_skip_special_tokens": ("rollout", "rollout_skip_special_tokens"),
    "rollout_stop": ("rollout", "rollout_stop"),
    "rollout_stop_token_ids": ("rollout", "rollout_stop_token_ids"),
    "rollout_shuffle": ("rollout", "rollout_shuffle"),
    "rollout_seed": ("rollout", "rollout_seed"),
    "over_sampling_batch_size": ("rollout", "over_sampling_batch_size"),
    "dynamic_sampling_filter_path": ("rollout", "dynamic_sampling_filter_path"),
    "partial_rollout": ("rollout", "partial_rollout"),
    "mask_offpolicy_in_partial_rollout": ("rollout", "mask_offpolicy_in_partial_rollout"),
    "custom_generate_function_path": ("rollout", "custom_generate_function_path"),
    "custom_rollout_log_function_path": ("rollout", "custom_rollout_log_function_path"),
    "custom_eval_rollout_log_function_path": ("rollout", "custom_eval_rollout_log_function_path"),
    "buffer_filter_path": ("rollout", "buffer_filter_path"),
    "update_weight_buffer_size": ("rollout", "update_weight_buffer_size"),
    "update_weights_interval": ("rollout", "update_weights_interval"),
    "keep_old_actor": ("rollout", "keep_old_actor"),
    "rollout_data_postprocess_path": ("rollout", "rollout_data_postprocess_path"),
    "rollout_external": ("rollout", "rollout_external"),
    "rollout_external_engine_addrs": ("rollout", "rollout_external_engine_addrs"),
    # Fault tolerance
    "use_fault_tolerance": ("fault_tolerance", "use_fault_tolerance"),
    "rollout_health_check_interval": ("fault_tolerance", "rollout_health_check_interval"),
    "rollout_health_check_timeout": ("fault_tolerance", "rollout_health_check_timeout"),
    "rollout_health_check_first_wait": ("fault_tolerance", "rollout_health_check_first_wait"),
    # Data
    "num_rollout": ("data", "num_rollout"),
    "num_epoch": ("data", "num_epoch"),
    "rollout_global_dataset": ("data", "rollout_global_dataset"),
    "data_source_path": ("data", "data_source_path"),
    "prompt_data": ("data", "prompt_data"),
    "apply_chat_template": ("data", "apply_chat_template"),
    "apply_chat_template_kwargs": ("data", "apply_chat_template_kwargs"),
    "input_key": ("data", "input_key"),
    "label_key": ("data", "label_key"),
    "metadata_key": ("data", "metadata_key"),
    "tool_key": ("data", "tool_key"),
    "multimodal_keys": ("data", "multimodal_keys"),
    "start_rollout_id": ("data", "start_rollout_id"),
    "rollout_batch_size": ("data", "rollout_batch_size"),
    "n_samples_per_prompt": ("data", "n_samples_per_prompt"),
    "global_batch_size": ("data", "global_batch_size"),
    "num_steps_per_rollout": ("data", "num_steps_per_rollout"),
    "micro_batch_size": ("data", "micro_batch_size"),
    "balance_data": ("data", "balance_data"),
    "use_dynamic_batch_size": ("data", "use_dynamic_batch_size"),
    "max_tokens_per_gpu": ("data", "max_tokens_per_gpu"),
    "log_probs_max_tokens_per_gpu": ("data", "log_probs_max_tokens_per_gpu"),
    # Evaluation
    "eval_function_path": ("evaluation", "eval_function_path"),
    "eval_interval": ("evaluation", "eval_interval"),
    "skip_eval_before_train": ("evaluation", "skip_eval_before_train"),
    "eval_prompt_data": ("evaluation", "eval_prompt_data"),
    "eval_config": ("evaluation", "eval_config"),
    "eval_input_key": ("evaluation", "eval_input_key"),
    "eval_label_key": ("evaluation", "eval_label_key"),
    "eval_tool_key": ("evaluation", "eval_tool_key"),
    "n_samples_per_eval_prompt": ("evaluation", "n_samples_per_eval_prompt"),
    "eval_temperature": ("evaluation", "eval_temperature"),
    "eval_top_p": ("evaluation", "eval_top_p"),
    "eval_top_k": ("evaluation", "eval_top_k"),
    "eval_max_response_len": ("evaluation", "eval_max_response_len"),
    "eval_max_prompt_len": ("evaluation", "eval_max_prompt_len"),
    "eval_min_new_tokens": ("evaluation", "eval_min_new_tokens"),
    "eval_max_context_len": ("evaluation", "eval_max_context_len"),
    # Algorithm
    "ref_load": ("algorithm", "ref_load"),
    "ref_ckpt_step": ("algorithm", "ref_ckpt_step"),
    "load": ("algorithm", "load"),
    "save": ("algorithm", "save"),
    "save_interval": ("algorithm", "save_interval"),
    "async_save": ("algorithm", "async_save"),
    "no_save_optim": ("algorithm", "no_save_optim"),
    "save_hf": ("algorithm", "save_hf"),
    "seed": ("algorithm", "seed"),
    "clip_grad": ("algorithm", "clip_grad"),
    "calculate_per_token_loss": ("algorithm", "calculate_per_token_loss"),
    "lr": ("algorithm", "lr"),
    "num_critic_only_steps": ("algorithm", "num_critic_only_steps"),
    "critic_load": ("algorithm", "critic_load"),
    "critic_save": ("algorithm", "critic_save"),
    "critic_lr": ("algorithm", "critic_lr"),
    "critic_lr_warmup_iters": ("algorithm", "critic_lr_warmup_iters"),
    "eps_clip": ("algorithm", "eps_clip"),
    "eps_clip_high": ("algorithm", "eps_clip_high"),
    "eps_clip_c": ("algorithm", "eps_clip_c"),
    "value_clip": ("algorithm", "value_clip"),
    "kl_coef": ("algorithm", "kl_coef"),
    "kl_loss_coef": ("algorithm", "kl_loss_coef"),
    "use_kl_loss": ("algorithm", "use_kl_loss"),
    "kl_loss_type": ("algorithm", "kl_loss_type"),
    "use_unbiased_kl": ("algorithm", "use_unbiased_kl"),
    "ref_update_interval": ("algorithm", "ref_update_interval"),
    "loss_type": ("algorithm", "loss_type"),
    "custom_loss_function_path": ("algorithm", "custom_loss_function_path"),
    "advantage_estimator": ("algorithm", "advantage_estimator"),
    "compute_advantages_and_returns": ("algorithm", "compute_advantages_and_returns"),
    "gamma": ("algorithm", "gamma"),
    "lambd": ("algorithm", "lambd"),
    "normalize_advantages": ("algorithm", "normalize_advantages"),
    "grpo_std_normalization": ("algorithm", "grpo_std_normalization"),
    "rewards_normalization": ("algorithm", "rewards_normalization"),
    "entropy_coef": ("algorithm", "entropy_coef"),
    "use_rollout_entropy": ("algorithm", "use_rollout_entropy"),
    "get_mismatch_metrics": ("algorithm", "get_mismatch_metrics"),
    "reset_optimizer_states": ("algorithm", "reset_optimizer_states"),
    "use_rollout_logprobs": ("algorithm", "use_rollout_logprobs"),
    "use_tis": ("algorithm", "use_tis"),
    "tis_clip": ("algorithm", "tis_clip"),
    "tis_clip_low": ("algorithm", "tis_clip_low"),
    "custom_tis_function_path": ("algorithm", "custom_tis_function_path"),
    "custom_pg_loss_reducer_function_path": ("algorithm", "custom_pg_loss_reducer_function_path"),
    "use_routing_replay": ("algorithm", "use_routing_replay"),
    "use_rollout_routing_replay": ("algorithm", "use_rollout_routing_replay"),
    "use_opsm": ("algorithm", "use_opsm"),
    "opsm_delta": ("algorithm", "opsm_delta"),
    # Router
    "use_slime_router": ("router", "use_slime_router"),
    "slime_router_middleware_paths": ("router", "slime_router_middleware_paths"),
    "slime_router_timeout": ("router", "slime_router_timeout"),
    "slime_router_max_connections": ("router", "slime_router_max_connections"),
    "slime_router_health_check_failure_threshold": ("router", "slime_router_health_check_failure_threshold"),
    # Wandb
    "use_wandb": ("wandb", "use_wandb"),
    "wandb_mode": ("wandb", "wandb_mode"),
    "wandb_dir": ("wandb", "wandb_dir"),
    "wandb_key": ("wandb", "wandb_key"),
    "wandb_host": ("wandb", "wandb_host"),
    "wandb_team": ("wandb", "wandb_team"),
    "wandb_group": ("wandb", "wandb_group"),
    "wandb_project": ("wandb", "wandb_project"),
    "wandb_random_suffix": ("wandb", "wandb_random_suffix"),
    "wandb_always_use_train_step": ("wandb", "wandb_always_use_train_step"),
    "wandb_run_id": ("wandb", "wandb_run_id"),
    "log_multi_turn": ("wandb", "log_multi_turn"),
    "log_passrate": ("wandb", "log_passrate"),
    "log_reward_category": ("wandb", "log_reward_category"),
    "log_correct_samples": ("wandb", "log_correct_samples"),
    # Tensorboard
    "use_tensorboard": ("tensorboard", "use_tensorboard"),
    "tb_project_name": ("tensorboard", "tb_project_name"),
    "tb_experiment_name": ("tensorboard", "tb_experiment_name"),
    # Debug
    "save_debug_rollout_data": ("debug", "save_debug_rollout_data"),
    "load_debug_rollout_data": ("debug", "load_debug_rollout_data"),
    "load_debug_rollout_data_subsample": ("debug", "load_debug_rollout_data_subsample"),
    "debug_rollout_only": ("debug", "debug_rollout_only"),
    "debug_train_only": ("debug", "debug_train_only"),
    "save_debug_train_data": ("debug", "save_debug_train_data"),
    "dump_details": ("debug", "dump_details"),
    "memory_snapshot_dir": ("debug", "memory_snapshot_dir"),
    "memory_snapshot_num_steps": ("debug", "memory_snapshot_num_steps"),
    "profile_target": ("debug", "profile_target"),
    "memory_recorder": ("debug", "memory_recorder"),
    "check_weight_update_equal": ("debug", "check_weight_update_equal"),
    # Network
    "http_proxy": ("network", "http_proxy"),
    "use_distributed_post": ("network", "use_distributed_post"),
    # Reward
    "rm_type": ("reward", "rm_type"),
    "reward_key": ("reward", "reward_key"),
    "eval_reward_key": ("reward", "eval_reward_key"),
    "group_rm": ("reward", "group_rm"),
    "rm_url": ("reward", "rm_url"),
    "custom_rm_path": ("reward", "custom_rm_path"),
    "custom_reward_post_process_path": ("reward", "custom_reward_post_process_path"),
    "custom_convert_samples_to_train_data_path": ("reward", "custom_convert_samples_to_train_data_path"),
    # Rollout buffer
    "rollout_buffer_url": ("rollout_buffer", "rollout_buffer_url"),
    "fetch_trajectory_retry_times": ("rollout_buffer", "fetch_trajectory_retry_times"),
    "min_batch_collection_ratio": ("rollout_buffer", "min_batch_collection_ratio"),
    "rollout_task_type": ("rollout_buffer", "rollout_task_type"),
    "loss_mask_type": ("rollout_buffer", "loss_mask_type"),
    "data_pad_size_multiplier": ("rollout_buffer", "data_pad_size_multiplier"),
    "rollout_sample_filter_path": ("rollout_buffer", "rollout_sample_filter_path"),
    "rollout_all_samples_process_path": ("rollout_buffer", "rollout_all_samples_process_path"),
    "disable_rollout_trim_samples": ("rollout_buffer", "disable_rollout_trim_samples"),
    "use_dynamic_global_batch_size": ("rollout_buffer", "use_dynamic_global_batch_size"),
    # Megatron plugins
    "custom_megatron_init_path": ("megatron_plugins", "custom_megatron_init_path"),
    "custom_megatron_before_log_prob_hook_path": ("megatron_plugins", "custom_megatron_before_log_prob_hook_path"),
    "custom_megatron_before_train_step_hook_path": ("megatron_plugins", "custom_megatron_before_train_step_hook_path"),
    # MTP
    "mtp_num_layers": ("mtp", "mtp_num_layers"),
    "mtp_loss_scaling_factor": ("mtp", "mtp_loss_scaling_factor"),
    "enable_mtp_training": ("mtp", "enable_mtp_training"),
    # Prefill/decode
    "prefill_num_servers": ("prefill_decode", "prefill_num_servers"),
    # CI
    "ci_test": ("ci", "ci_test"),
    "ci_disable_kl_checker": ("ci", "ci_disable_kl_checker"),
    "ci_metric_checker_key": ("ci", "ci_metric_checker_key"),
    "ci_metric_checker_threshold": ("ci", "ci_metric_checker_threshold"),
    "ci_save_grad_norm": ("ci", "ci_save_grad_norm"),
    "ci_load_grad_norm": ("ci", "ci_load_grad_norm"),
    # SGLang
    "sglang_tensor_parallel_size": ("sglang", "sglang_tensor_parallel_size"),
    "sglang_mem_fraction_static": ("sglang", "sglang_mem_fraction_static"),
    "sglang_pp_size": ("sglang", "sglang_pp_size"),
    "sglang_pipeline_parallel_size": ("sglang", "sglang_pipeline_parallel_size"),
}


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
    Unknown args are placed at the root level (for Megatron passthrough).
    """
    result: dict[str, Any] = {}

    for key, value in flat.items():
        if value is None:
            # Skip None values to allow YAML defaults
            continue

        if key in ARG_TO_SECTION:
            section, field = ARG_TO_SECTION[key]
            if section not in result:
                result[section] = {}
            result[section][field] = value
        else:
            # Put unmapped args at root level (Megatron passthrough)
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
