#!/usr/bin/env python3
"""
GRPO Training Launcher with Compute YAML Support.

This launcher enables running training from a compute configuration YAML file
that contains training parameters in a `parameters:` section. This format is
commonly used with cluster schedulers like MCLI.

Usage:
    # Launch with compute YAML (extracts parameters section)
    python launch_grpo.py -f /mnt/config/parameters.yaml

    # Launch with multiple config files (merged in order)
    python launch_grpo.py -f base.yaml -f override.yaml

    # Launch with CLI overrides
    python launch_grpo.py -f config.yaml --optimizer.lr=1e-7

Example compute YAML format:
    name: my-training-run
    image: mosaicml/pytorch:latest
    compute:
      gpus: 16
    command: |-
      python launch_grpo.py -f /mnt/config/parameters.yaml
    parameters:
      pretrain_model_name: Qwen/Qwen3-4B
      max_gen_len: 8192
      ...
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch.distributed as dist
from omegaconf import OmegaConf, DictConfig

logger = logging.getLogger(__name__)


def setup_logging(rank: int = 0):
    """Configure logging with rank prefix."""
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def load_compute_yaml(yaml_path: str) -> DictConfig:
    """
    Load a compute YAML file and extract the parameters section.

    Handles both formats:
    1. Compute YAML with 'parameters:' section (extracts parameters)
    2. Plain training config (uses as-is)

    Args:
        yaml_path: Path to YAML file

    Returns:
        OmegaConf DictConfig with training parameters
    """
    config = OmegaConf.load(yaml_path)

    # Check if this is a compute YAML (has 'parameters' key)
    if "parameters" in config:
        logger.info(f"Detected compute YAML format, extracting 'parameters' section from {yaml_path}")
        return config.parameters
    else:
        # Plain training config
        logger.info(f"Using config as-is from {yaml_path}")
        return config


def merge_configs(config_files: List[str], cli_overrides: List[str] = None) -> DictConfig:
    """
    Merge multiple config files and CLI overrides.

    Files are merged in order (later files override earlier ones).
    CLI overrides are applied last.

    Args:
        config_files: List of YAML file paths
        cli_overrides: List of CLI overrides in dot notation (e.g., ["optimizer.lr=1e-7"])

    Returns:
        Merged OmegaConf DictConfig
    """
    merged_config = OmegaConf.create({})

    for config_file in config_files:
        if not Path(config_file).exists():
            raise FileNotFoundError(f"Config file not found: {config_file}")

        file_config = load_compute_yaml(config_file)
        merged_config = OmegaConf.merge(merged_config, file_config)
        logger.info(f"Merged config from: {config_file}")

    # Apply CLI overrides
    if cli_overrides:
        cli_config = OmegaConf.from_dotlist(cli_overrides)
        merged_config = OmegaConf.merge(merged_config, cli_config)
        logger.info(f"Applied CLI overrides: {cli_overrides}")

    # Resolve interpolations (e.g., ${pretrain_model_name})
    OmegaConf.resolve(merged_config)

    return merged_config


def config_to_args_list(config: DictConfig, prefix: str = "") -> List[str]:
    """
    Convert OmegaConf config to list of CLI arguments.

    Handles nested configs by flattening with appropriate arg names.

    Args:
        config: OmegaConf DictConfig
        prefix: Prefix for nested keys

    Returns:
        List of CLI arguments (e.g., ["--lr", "1e-6", "--use-mlflow"])
    """
    args_list = []

    for key, value in config.items():
        if isinstance(value, DictConfig):
            # Recursively handle nested configs
            # Some nested configs map to specific arg patterns
            nested_args = config_to_args_list(value, prefix=key)
            args_list.extend(nested_args)
        elif value is None:
            continue
        elif isinstance(value, bool):
            if value:
                args_list.append(f"--{key.replace('_', '-')}")
        else:
            args_list.extend([f"--{key.replace('_', '-')}", str(value)])

    return args_list


def get_keys_to_skip() -> set:
    """
    Get keys that should be skipped (not passed to slime).

    These are compute YAML metadata keys that are not training parameters.
    """
    return {
        # Compute YAML metadata (used by MCLI, not slime)
        "name", "image", "compute", "scheduling", "integrations", "command",
    }


# =============================================================================
# Parameter Name Reference (for YAML config authors)
# =============================================================================
# The YAML config should use slime's native argument names directly.
# Here's a reference mapping common aliases to slime argument names:
#
# Model:
#   pretrain_model_name  ->  hf_checkpoint
#
# Generation:
#   max_gen_len          ->  rollout_max_response_len
#   max_model_len        ->  rollout_max_context_len
#   generations_per_prompt -> n_samples_per_prompt
#
# Training:
#   global_train_batch_size -> global_batch_size
#   max_duration         ->  num_rollout
#
# Data:
#   data_path            ->  prompt_data
#   prompt_key           ->  input_key
#   response_key         ->  label_key
#
# See slime/utils/arguments.py for the full list of available arguments.
# =============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch GRPO training with compute YAML config support",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    parser.add_argument(
        "-f", "--config-file",
        action="append",
        dest="config_files",
        default=[],
        help="Config file path (can be specified multiple times for merging)",
    )

    parser.add_argument(
        "--launch-method",
        type=str,
        default="torchrun",
        choices=["torchrun", "ray_job_submit"],
        help="Launch method (default: torchrun)",
    )

    parser.add_argument(
        "--train-backend",
        type=str,
        default="fsdp",
        choices=["fsdp", "megatron"],
        help="Training backend (default: fsdp)",
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print config without launching training",
    )

    # Capture remaining args as CLI overrides
    args, remaining = parser.parse_known_args()

    # Convert remaining args to overrides (handle --key=value and --key value formats)
    cli_overrides = []
    i = 0
    while i < len(remaining):
        arg = remaining[i]
        if arg.startswith("--"):
            if "=" in arg:
                # --key=value format
                cli_overrides.append(arg[2:])  # Remove --
            elif i + 1 < len(remaining) and not remaining[i + 1].startswith("--"):
                # --key value format
                key = arg[2:]
                value = remaining[i + 1]
                cli_overrides.append(f"{key}={value}")
                i += 1
            else:
                # Boolean flag
                cli_overrides.append(f"{arg[2:]}=true")
        i += 1

    args.cli_overrides = cli_overrides

    return args


def main():
    """Main entry point for the GRPO launcher."""
    args = parse_args()

    # Setup logging
    rank = int(os.environ.get("RANK", 0))
    setup_logging(rank)

    logger.info("=== GRPO Training Launcher ===")
    logger.info(f"Config files: {args.config_files}")
    logger.info(f"Launch method: {args.launch_method}")
    logger.info(f"Train backend: {args.train_backend}")

    if not args.config_files:
        logger.error("No config files specified. Use -f/--config-file to specify config files.")
        sys.exit(1)

    # Load and merge configs
    config = merge_configs(args.config_files, args.cli_overrides)

    # Build a clean config with only slime-compatible keys
    # (skip compute YAML metadata and nested configs like 'loggers')
    keys_to_skip = get_keys_to_skip()
    slime_config = OmegaConf.create({})

    for key, value in config.items():
        # Skip nested configs like 'loggers' - handle separately
        if isinstance(value, DictConfig):
            continue
        # Skip compute YAML metadata keys
        if key in keys_to_skip:
            logger.info(f"Skipping compute YAML metadata key: {key}")
            continue
        # Include all other keys directly (no mapping)
        slime_config[key] = value

    logger.info(f"Loaded {len(slime_config)} training parameters from config")

    if args.dry_run:
        logger.info("=== Dry Run - Config Summary ===")
        print(OmegaConf.to_yaml(slime_config))
        return

    # Save merged config for reference
    merged_config_path = "/tmp/merged_training_config.yaml"
    OmegaConf.save(slime_config, merged_config_path)
    logger.info(f"Saved merged config to: {merged_config_path}")

    # Build CLI args list from the mapped parameters
    cli_args_from_config = []
    for key, value in slime_config.items():
        if value is None:
            continue
        arg_name = f"--{key.replace('_', '-')}"
        if isinstance(value, bool):
            if value:
                cli_args_from_config.append(arg_name)
        else:
            cli_args_from_config.extend([arg_name, str(value)])

    # Extract MLflow settings if present
    mlflow_args = []
    if "loggers" in config and "mlflow" in config.loggers:
        mlflow_config = config.loggers.mlflow
        mlflow_args = [
            "--use-mlflow",
            "--mlflow-tracking-uri", mlflow_config.get("tracking_uri", "databricks"),
            "--mlflow-experiment-name", mlflow_config.get("experiment_name", "slime_training"),
        ]
        if "tags" in mlflow_config:
            if "group" in mlflow_config.tags:
                mlflow_args.extend(["--mlflow-group", mlflow_config.tags.group])

    # Launch training
    if args.launch_method == "torchrun":
        from slime.utils.arguments import parse_args as slime_parse_args

        # Build args for slime - use CLI args directly instead of --config
        slime_args_list = [
            "--launch-method", "torchrun",
            "--train-backend", args.train_backend,
        ]
        slime_args_list.extend(cli_args_from_config)
        slime_args_list.extend(mlflow_args)

        logger.info(f"Slime args: {slime_args_list}")

        # Parse slime args
        sys.argv = ["train.py"] + slime_args_list
        training_args = slime_parse_args()

        # Check if we're already in a torchrun environment (RANK is set)
        if os.environ.get("RANK") is not None:
            # Running via torchrun - use SPMD launcher with torch.distributed
            from slime.launcher.spmd_ray_bridge import start_ray_server
            with start_ray_server() as ray_address:
                logger.info(f"Ray cluster ready at {ray_address}")

                if dist.get_rank() == 0:
                    from train import train
                    train(training_args)

                logger.info("Training complete")
        else:
            # Running directly (not via torchrun) - just start Ray and run training
            logger.info("Not running via torchrun, starting Ray directly")
            import ray
            if not ray.is_initialized():
                ray.init()
            logger.info(f"Ray initialized: {ray.cluster_resources()}")

            from train import train
            train(training_args)

            logger.info("Training complete")
    else:
        # Ray Job Submit mode - just parse args and call train
        from slime.utils.arguments import parse_args as slime_parse_args
        from train import train

        slime_args_list = [
            "--train-backend", args.train_backend,
        ]
        slime_args_list.extend(cli_args_from_config)
        slime_args_list.extend(mlflow_args)

        logger.info(f"Slime args: {slime_args_list}")

        sys.argv = ["train.py"] + slime_args_list
        training_args = slime_parse_args()
        train(training_args)


if __name__ == "__main__":
    main()
