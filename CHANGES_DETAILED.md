# CHANGES_DETAILED.md - Implementation Guide

This document provides a step-by-step implementation guide for the goals outlined in CHANGES.md. The implementation is ordered to minimize dependencies and maximize rebase-friendliness.

## Table of Contents

1. [Goals Overview](#goals-overview)
2. [Implementation Order](#implementation-order)
3. [Phase 1: OmegaConf YAML Configuration System](#phase-1-omegaconf-yaml-configuration-system)
4. [Phase 2: MLflow Logging Support](#phase-2-mlflow-logging-support)
5. [Phase 3: SPMD-style Launch Support](#phase-3-spmd-style-torchrun-launch-support)
6. [Phase 4: Integration Testing](#phase-4-integration-testing)
7. [File Summary](#file-summary)

---

## Goals Overview

1. **SPMD-style Launch Support** - Enable torchrun-based multi-node training alongside existing Ray Job Submit
2. **MLflow Logging** - Add MLflow as a logging backend alongside WandB and TensorBoard
3. **OmegaConf YAML Configs** - Migrate from bash script configs to structured YAML configs with variable interpolation

---

## Implementation Order

**Recommended Order: OmegaConf Configs → MLflow Logging → SPMD Launch**

Rationale:
- OmegaConf configs provide foundational infrastructure that benefits other features
- MLflow logging is independent and simpler, good for incremental progress
- SPMD launch is most complex, benefits from having config system in place

---

## Phase 1: OmegaConf YAML Configuration System

### Step 1.1: Create Config Module Structure

Create the directory structure:

```bash
mkdir -p slime/config
mkdir -p configs/training
```

### Step 1.2: Create Config Module Init

**File: `slime/config/__init__.py`**

```python
"""Slime configuration module using OmegaConf."""

from .loader import load_config, load_yaml_as_dict
from .base import SlimeConfig
from .validation import validate_config
from .converter import config_to_namespace, namespace_to_config

__all__ = [
    "load_config",
    "load_yaml_as_dict",
    "SlimeConfig",
    "validate_config",
    "config_to_namespace",
    "namespace_to_config",
]
```

### Step 1.3: Implement Config Dataclasses

**File: `slime/config/base.py`**

```python
"""Configuration dataclasses for Slime training."""

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any


@dataclass
class ClusterConfig:
    """Ray cluster and GPU allocation settings."""
    actor_num_nodes: int = 1
    actor_num_gpus_per_node: int = 8
    critic_num_nodes: Optional[int] = None
    critic_num_gpus_per_node: Optional[int] = None
    rollout_num_gpus: int = 8
    rollout_num_gpus_per_engine: int = 1
    colocate: bool = False
    colocate_actor_ref: bool = False
    colocate_critic_ref: bool = False
    offload_actor: bool = False
    offload_reference_model: bool = False


@dataclass
class CheckpointConfig:
    """Checkpoint loading and saving settings."""
    hf_checkpoint: Optional[str] = None
    load: Optional[str] = None
    save: Optional[str] = None
    ref_load: Optional[str] = None
    save_interval: int = 100


@dataclass
class RolloutConfig:
    """Rollout and generation settings."""
    prompt_data: Optional[str] = None
    input_key: str = "prompt"
    label_key: Optional[str] = None
    tool_key: Optional[str] = None
    apply_chat_template: bool = False
    shuffle: bool = True
    temperature: float = 1.0
    top_p: float = 1.0
    max_response_len: int = 4096
    max_context_len: int = 8192
    batch_size: int = 32
    n_samples_per_prompt: int = 1
    partial_rollout: bool = False


@dataclass
class DataConfig:
    """Data loading and batching settings."""
    global_batch_size: int = 256
    balance_data: bool = True
    use_dynamic_batch_size: bool = False
    max_tokens_per_gpu: Optional[int] = None
    seed: int = 1234


@dataclass
class EvalConfig:
    """Evaluation settings."""
    interval: int = 100
    prompt_data: Optional[str] = None
    config_path: Optional[str] = None
    n_samples_per_eval_prompt: int = 1
    max_response_len: int = 4096
    top_p: float = 1.0


@dataclass
class AlgorithmConfig:
    """PPO/GRPO algorithm settings."""
    advantage_estimator: str = "grpo"
    eps_clip: float = 0.2
    eps_clip_high: Optional[float] = None
    kl_coef: float = 0.0
    use_kl_loss: bool = False
    kl_loss_coef: float = 0.0
    kl_loss_type: str = "low_var_kl"
    entropy_coef: float = 0.0
    gamma: float = 1.0
    lam: float = 1.0


@dataclass
class OptimizerConfig:
    """Optimizer settings."""
    name: str = "adam"
    lr: float = 1e-6
    lr_decay_style: str = "constant"
    weight_decay: float = 0.1
    adam_beta1: float = 0.9
    adam_beta2: float = 0.98
    adam_eps: float = 1e-8
    clip_grad: float = 1.0


@dataclass
class RewardConfig:
    """Reward model settings."""
    rm_type: str = "deepscaler"
    custom_rm_path: Optional[str] = None
    rm_url: Optional[str] = None


@dataclass
class SGLangConfig:
    """SGLang inference server settings."""
    mem_fraction_static: float = 0.7
    chunked_prefill_size: int = 8192
    disable_flashinfer: bool = False
    disable_radix_cache: bool = False


@dataclass
class WandbConfig:
    """Weights & Biases logging settings."""
    enabled: bool = False
    project: Optional[str] = None
    group: Optional[str] = None
    key: Optional[str] = None
    mode: str = "online"


@dataclass
class TensorboardConfig:
    """TensorBoard logging settings."""
    enabled: bool = False
    project_name: Optional[str] = None
    experiment_name: Optional[str] = None


@dataclass
class MLflowConfig:
    """MLflow logging settings."""
    enabled: bool = False
    tracking_uri: Optional[str] = None
    experiment_name: Optional[str] = None
    run_name: Optional[str] = None
    group: Optional[str] = None
    tags: Optional[Dict[str, str]] = None


@dataclass
class LoggingConfig:
    """Combined logging settings."""
    wandb: WandbConfig = field(default_factory=WandbConfig)
    tensorboard: TensorboardConfig = field(default_factory=TensorboardConfig)
    mlflow: MLflowConfig = field(default_factory=MLflowConfig)


@dataclass
class TrainConfig:
    """Training backend settings."""
    backend: str = "megatron"  # "megatron" or "fsdp"
    num_rollout: int = 1000
    distributed_timeout_minutes: int = 10
    env_vars: Optional[Dict[str, str]] = None


@dataclass
class FSDPConfig:
    """FSDP-specific settings."""
    gradient_checkpointing: bool = False
    attn_implementation: str = "flash_attention_2"
    cpu_offload: bool = False
    state_dict_cpu_offload: bool = True


@dataclass
class MegatronConfig:
    """Megatron-specific settings."""
    tensor_model_parallel_size: int = 1
    pipeline_model_parallel_size: int = 1
    context_parallel_size: int = 1
    sequence_parallel: bool = False
    recompute_granularity: Optional[str] = None
    recompute_method: Optional[str] = None
    recompute_num_layers: Optional[int] = None


@dataclass
class ModelConfig:
    """Model architecture settings."""
    num_layers: Optional[int] = None
    hidden_size: Optional[int] = None
    ffn_hidden_size: Optional[int] = None
    num_attention_heads: Optional[int] = None
    num_query_groups: Optional[int] = None
    vocab_size: Optional[int] = None
    kv_channels: Optional[int] = None
    swiglu: bool = False
    group_query_attention: bool = False
    use_rotary_position_embeddings: bool = True
    disable_bias_linear: bool = False
    normalization: str = "RMSNorm"
    norm_epsilon: float = 1e-6
    rotary_base: int = 10000
    qk_layernorm: bool = False


@dataclass
class LauncherConfig:
    """Launch method settings."""
    method: str = "ray_job_submit"  # "ray_job_submit" or "torchrun"
    spmd_ray_timeout: int = 300


@dataclass
class SlimeConfig:
    """Root configuration for Slime training."""
    # Global settings
    seed: int = 1234
    pretrain_model_name: Optional[str] = None
    max_gen_len: int = 8192
    max_model_len: int = 10240

    # Sub-configs
    cluster: ClusterConfig = field(default_factory=ClusterConfig)
    checkpoint: CheckpointConfig = field(default_factory=CheckpointConfig)
    rollout: RolloutConfig = field(default_factory=RolloutConfig)
    data: DataConfig = field(default_factory=DataConfig)
    eval: EvalConfig = field(default_factory=EvalConfig)
    algorithm: AlgorithmConfig = field(default_factory=AlgorithmConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)
    sglang: SGLangConfig = field(default_factory=SGLangConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    train: TrainConfig = field(default_factory=TrainConfig)
    fsdp: FSDPConfig = field(default_factory=FSDPConfig)
    megatron: MegatronConfig = field(default_factory=MegatronConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    launcher: LauncherConfig = field(default_factory=LauncherConfig)
```

### Step 1.4: Implement OmegaConf Loader

**File: `slime/config/loader.py`**

```python
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
) -> Dict[str, Any]:
    """
    Flatten nested config dictionary for argparse compatibility.

    Converts nested keys to flat keys:
        {'optimizer': {'lr': 1e-6}} -> {'optimizer_lr': 1e-6}

    Also creates shorthand keys without prefix for top-level access:
        {'optimizer': {'lr': 1e-6}} -> {'lr': 1e-6, 'optimizer_lr': 1e-6}
    """
    items = []
    for key, value in config.items():
        new_key = f"{parent_key}{sep}{key}" if parent_key else key

        if isinstance(value, dict):
            # Recursively flatten nested dicts
            items.extend(_flatten_config(value, new_key, sep=sep).items())
            # Also add shorthand keys for leaf values
            for nested_key, nested_value in value.items():
                if not isinstance(nested_value, dict):
                    items.append((nested_key.replace("-", "_"), nested_value))
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
```

### Step 1.5: Implement Config Converter

**File: `slime/config/converter.py`**

```python
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
```

### Step 1.6: Implement Config Validation

**File: `slime/config/validation.py`**

```python
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
```

### Step 1.7: Integrate with Existing Arguments.py

**Modify: `slime/utils/arguments.py`**

Add these changes to integrate YAML config loading:

```python
# At the top of the file, add import:
import sys

# Add new function before parse_args():
def _extract_explicit_args(argv=None):
    """Extract which arguments were explicitly provided on command line."""
    if argv is None:
        argv = sys.argv[1:]

    explicit = set()
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            # Extract argument name
            arg_name = arg[2:].split("=")[0].replace("-", "_")
            explicit.add(arg_name)
        i += 1
    return explicit


# Modify parse_args() - add at the beginning of the function:
def parse_args(add_custom_arguments=None):
    # NEW: Early parse to check for --config
    pre_parser = argparse.ArgumentParser(add_help=False)
    pre_parser.add_argument("--config", type=str, default=None,
        help="Path to YAML config file. CLI args override YAML values.")
    pre_args, remaining_argv = pre_parser.parse_known_args()

    # Track which args were explicitly provided
    explicit_args = _extract_explicit_args()

    # NEW: Load YAML defaults if config provided
    yaml_defaults = {}
    if pre_args.config:
        try:
            from slime.config.loader import load_yaml_as_dict
            yaml_defaults = load_yaml_as_dict(pre_args.config)
            print(f"Loaded config from {pre_args.config}")
        except Exception as e:
            print(f"Warning: Failed to load config {pre_args.config}: {e}")

    # ... existing code ...

    # After args are parsed, apply YAML defaults for unset values:
    if yaml_defaults:
        from slime.config.converter import apply_yaml_defaults_to_namespace
        args = apply_yaml_defaults_to_namespace(args, yaml_defaults, explicit_args)

    # Store config path in args for reference
    args.config = pre_args.config

    return args
```

### Step 1.8: Create Example YAML Config

**File: `configs/training/qwen3-4b-grpo.yaml`**

```yaml
# Slime Training Configuration
# Model: Qwen3-4B with GRPO algorithm
#
# Usage:
#   python train.py --config configs/training/qwen3-4b-grpo.yaml
#   python train.py --config configs/training/qwen3-4b-grpo.yaml --lr 1e-7  # Override

parameters:
  # Global settings with variable interpolation
  seed: 1234
  pretrain_model_name: /root/Qwen3-4B
  max_gen_len: 8192
  max_model_len: 10240

  # Cluster settings
  cluster:
    actor_num_nodes: 1
    actor_num_gpus_per_node: 2
    rollout_num_gpus: 2
    rollout_num_gpus_per_engine: 1
    colocate: false

  # Checkpoint settings
  checkpoint:
    hf_checkpoint: ${pretrain_model_name}
    ref_load: ${pretrain_model_name}_torch_dist
    load: ${pretrain_model_name}_slime/
    save: ${pretrain_model_name}_slime/
    save_interval: 20

  # Training backend
  train:
    backend: megatron
    num_rollout: 3000
    env_vars:
      PYTORCH_CUDA_ALLOC_CONF: "expandable_segments:True"

  # Rollout settings
  rollout:
    prompt_data: /root/dapo-math-17k/dapo-math-17k.jsonl
    input_key: prompt
    label_key: label
    apply_chat_template: true
    shuffle: true
    temperature: 1.0
    max_response_len: ${max_gen_len}
    max_context_len: ${max_model_len}
    batch_size: 32
    n_samples_per_prompt: 8

  # Data settings
  data:
    global_batch_size: 256
    balance_data: true
    use_dynamic_batch_size: true
    max_tokens_per_gpu: 9216
    seed: ${seed}

  # Algorithm (GRPO)
  algorithm:
    advantage_estimator: grpo
    eps_clip: 0.2
    eps_clip_high: 0.28
    use_kl_loss: true
    kl_loss_coef: 0.0
    kl_loss_type: low_var_kl
    entropy_coef: 0.0

  # Optimizer
  optimizer:
    name: adam
    lr: 1e-6
    lr_decay_style: constant
    weight_decay: 0.1
    adam_beta1: 0.9
    adam_beta2: 0.98

  # Evaluation
  eval:
    interval: 20
    prompt_data: /root/aime-2024/aime-2024.jsonl
    n_samples_per_eval_prompt: 16
    max_response_len: 16384
    top_p: 1.0

  # Reward model
  reward:
    rm_type: deepscaler

  # SGLang settings
  sglang:
    mem_fraction_static: 0.7

  # Logging - all disabled by default
  logging:
    wandb:
      enabled: false
    tensorboard:
      enabled: false
    mlflow:
      enabled: false
      # tracking_uri: databricks
      # experiment_name: test_experiment

  # Model architecture (Qwen3-4B)
  model:
    swiglu: true
    num_layers: 36
    hidden_size: 2560
    ffn_hidden_size: 9728
    num_attention_heads: 32
    group_query_attention: true
    num_query_groups: 8
    use_rotary_position_embeddings: true
    disable_bias_linear: true
    normalization: RMSNorm
    norm_epsilon: 1e-6
    rotary_base: 1000000
    vocab_size: 151936
    kv_channels: 128
    qk_layernorm: true

  # Megatron-specific settings
  megatron:
    tensor_model_parallel_size: 2
    pipeline_model_parallel_size: 1
    sequence_parallel: true
    recompute_granularity: full
    recompute_method: uniform
    recompute_num_layers: 1

  # Launcher settings
  launcher:
    method: ray_job_submit  # or "torchrun"
    spmd_ray_timeout: 300
```

---

## Phase 2: MLflow Logging Support

### Step 2.1: Create MLflow Utils Module (using MlflowClient)

**File: `slime/utils/mlflow_utils.py`**

This implementation uses `mlflow.tracking.MlflowClient` for more fine-grained control
over experiment and run management, which is better suited for distributed training scenarios.

```python
"""
MLflow logging utilities for Slime distributed training.

Uses mlflow.tracking.MlflowClient for explicit control over experiments and runs.
This approach is preferred for distributed training because:
1. MlflowClient provides explicit experiment/run management without global state
2. Better suited for multi-process scenarios where each process needs to log to the same run
3. More control over tracking URI configuration per client instance
"""

import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
import uuid

logger = logging.getLogger(__name__)

# Global state for the current run
_mlflow_client: Optional["MlflowClient"] = None
_current_run_id: Optional[str] = None
_tracking_uri: Optional[str] = None


def _get_mlflow_client() -> "MlflowClient":
    """
    Get or create the MlflowClient singleton.

    Returns:
        MlflowClient instance configured with the current tracking URI
    """
    global _mlflow_client, _tracking_uri

    if _mlflow_client is None:
        try:
            from mlflow.tracking import MlflowClient
        except ImportError:
            raise ImportError(
                "mlflow is required for MLflow logging. "
                "Install with: pip install mlflow"
            )

        _mlflow_client = MlflowClient(tracking_uri=_tracking_uri)
        logger.info(f"Created MlflowClient with tracking_uri={_tracking_uri}")

    return _mlflow_client


def _set_tracking_uri(tracking_uri: Optional[str]) -> str:
    """
    Set the MLflow tracking URI.

    Handles special case for 'databricks' URI.

    Args:
        tracking_uri: The tracking URI or 'databricks'

    Returns:
        The resolved tracking URI
    """
    global _tracking_uri, _mlflow_client

    if tracking_uri:
        if tracking_uri.lower() == "databricks":
            _tracking_uri = "databricks"
        else:
            _tracking_uri = tracking_uri
    else:
        _tracking_uri = os.environ.get("MLFLOW_TRACKING_URI")

    # Reset client to pick up new URI
    _mlflow_client = None

    return _tracking_uri


def _flatten_dict(d: Dict[str, Any], parent_key: str = "", sep: str = ".") -> Dict[str, Any]:
    """
    Flatten nested dict for MLflow params.

    MLflow params don't support nested structures, so we flatten with dot notation.

    Example:
        {'optimizer': {'lr': 1e-6}} -> {'optimizer.lr': 1e-6}
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def _sanitize_metric_name(name: str) -> str:
    """
    Sanitize metric name for MLflow.

    MLflow allows: alphanumerics, underscores, dashes, periods, spaces, slashes.
    Replace any invalid characters.
    """
    # Most common names like "train/loss" are already valid
    return name.replace(":", "_")


def _compute_config_for_logging(args) -> Dict[str, Any]:
    """
    Prepare config dict for MLflow params logging.

    Filters to only include serializable values and adds environment context.
    """
    output = {}

    for k, v in vars(args).items():
        if isinstance(v, (str, int, float, bool, type(None))):
            output[k] = v

    # Add relevant environment variables for reproducibility
    whitelist_env_vars = [
        "SLURM_JOB_ID",
        "SLURM_JOB_NAME",
        "CUDA_VISIBLE_DEVICES",
        "WORLD_SIZE",
        "RANK",
        "LOCAL_RANK",
    ]
    for var in whitelist_env_vars:
        if var in os.environ:
            output[f"env.{var}"] = os.environ[var]

    return output


def _get_or_create_experiment(client: "MlflowClient", experiment_name: str) -> str:
    """
    Get existing experiment or create a new one.

    Args:
        client: MlflowClient instance
        experiment_name: Name of the experiment

    Returns:
        Experiment ID
    """
    experiment = client.get_experiment_by_name(experiment_name)
    if experiment is not None:
        logger.info(f"Using existing experiment: {experiment_name} (id={experiment.experiment_id})")
        return experiment.experiment_id

    # Create new experiment
    experiment_id = client.create_experiment(experiment_name)
    logger.info(f"Created new experiment: {experiment_name} (id={experiment_id})")
    return experiment_id


def init_mlflow_primary(args, **kwargs) -> None:
    """
    Initialize MLflow on the primary rank (rank 0 in main train.py).

    Uses MlflowClient to:
    - Set up tracking URI
    - Get or create the experiment
    - Create a new run with tags
    - Log initial params
    - Store run_id in args for secondary processes

    Args:
        args: Parsed arguments with MLflow configuration
        **kwargs: Additional keyword arguments (unused)
    """
    global _current_run_id

    if not getattr(args, "use_mlflow", False):
        args.mlflow_run_id = None
        return

    # Set tracking URI
    tracking_uri = getattr(args, "mlflow_tracking_uri", None)
    resolved_uri = _set_tracking_uri(tracking_uri)
    logger.info(f"MLflow tracking URI: {resolved_uri}")

    # Get client
    client = _get_mlflow_client()

    # Get or create experiment
    experiment_name = getattr(args, "mlflow_experiment_name", None)
    if not experiment_name:
        raise ValueError("--mlflow-experiment-name is required when using --use-mlflow")

    experiment_id = _get_or_create_experiment(client, experiment_name)

    # Prepare tags
    tags = {}
    mlflow_group = getattr(args, "mlflow_group", None)
    if mlflow_group:
        tags["group"] = mlflow_group
        tags["mlflow.runName"] = mlflow_group  # Sets display name in UI

    mlflow_tags = getattr(args, "mlflow_tags", None)
    if mlflow_tags:
        tags.update(mlflow_tags)

    # Add system tags
    tags["slime.version"] = "1.0"
    tags["slime.distributed"] = "true"

    # Generate run name
    run_name = getattr(args, "mlflow_run_name", None) or mlflow_group
    if run_name:
        run_name = f"{run_name}_{uuid.uuid4().hex[:6]}"
    else:
        run_name = f"slime_run_{uuid.uuid4().hex[:6]}"

    # Create run using MlflowClient
    run = client.create_run(
        experiment_id=experiment_id,
        run_name=run_name,
        tags=tags,
    )
    run_id = run.info.run_id
    _current_run_id = run_id

    logger.info(f"MLflow run created: {run_id} (name={run_name})")

    # Log params using client
    config = _compute_config_for_logging(args)
    flat_config = _flatten_dict(config)

    # MLflow has a limit on param value length (500 chars)
    # Log in batches to avoid issues with large param sets
    param_items = [(k, str(v)[:500]) for k, v in flat_config.items()]
    batch_size = 100

    for i in range(0, len(param_items), batch_size):
        batch = param_items[i:i + batch_size]
        for key, value in batch:
            try:
                client.log_param(run_id, key, value)
            except Exception as e:
                logger.warning(f"Failed to log param {key}: {e}")

    # Store run_id for secondary processes
    args.mlflow_run_id = run_id
    logger.info(f"MLflow primary initialized: run_id={run_id}")


def init_mlflow_secondary(args, **kwargs) -> None:
    """
    Initialize MLflow on secondary ranks (joins existing run).

    Secondary processes use the run_id from the primary to log to the same run.
    Uses MlflowClient for explicit run reference.

    Args:
        args: Parsed arguments with mlflow_run_id from primary
        **kwargs: Additional keyword arguments (unused)
    """
    global _current_run_id

    mlflow_run_id = getattr(args, "mlflow_run_id", None)
    if mlflow_run_id is None:
        return

    # Set tracking URI (same as primary)
    tracking_uri = getattr(args, "mlflow_tracking_uri", None)
    _set_tracking_uri(tracking_uri)

    # Store run_id for logging
    _current_run_id = mlflow_run_id

    # Verify run exists
    client = _get_mlflow_client()
    try:
        run = client.get_run(mlflow_run_id)
        logger.info(f"MLflow secondary joined run: {mlflow_run_id} (status={run.info.status})")
    except Exception as e:
        logger.warning(f"Could not verify run {mlflow_run_id}: {e}")


def mlflow_log_metrics(metrics: Dict[str, Any], step: int) -> None:
    """
    Log metrics to MLflow with step using MlflowClient.

    Handles:
    - Metric name sanitization
    - Type conversion (torch tensors to Python scalars)
    - Filtering non-numeric values
    - Batch logging for efficiency

    Args:
        metrics: Dictionary of metric names to values
        step: Training step number
    """
    global _current_run_id

    if _current_run_id is None:
        return

    client = _get_mlflow_client()
    timestamp = int(time.time() * 1000)  # MLflow expects milliseconds

    # Prepare metrics for batch logging
    from mlflow.entities import Metric

    metric_list: List[Metric] = []

    for k, v in metrics.items():
        # Convert value to float
        if isinstance(v, (int, float)):
            value = float(v)
        elif hasattr(v, "item"):  # torch tensor
            value = float(v.item())
        else:
            continue  # Skip non-numeric values

        metric_list.append(Metric(
            key=_sanitize_metric_name(k),
            value=value,
            timestamp=timestamp,
            step=step,
        ))

    if metric_list:
        try:
            client.log_batch(
                run_id=_current_run_id,
                metrics=metric_list,
            )
        except Exception as e:
            logger.warning(f"Failed to log metrics batch: {e}")
            # Fallback to individual logging
            for metric in metric_list:
                try:
                    client.log_metric(
                        run_id=_current_run_id,
                        key=metric.key,
                        value=metric.value,
                        step=metric.step,
                    )
                except Exception as e2:
                    logger.warning(f"Failed to log metric {metric.key}: {e2}")


def mlflow_log_param(key: str, value: Any) -> None:
    """
    Log a single parameter using MlflowClient.

    Args:
        key: Parameter name
        value: Parameter value (will be converted to string)
    """
    global _current_run_id

    if _current_run_id is None:
        return

    client = _get_mlflow_client()
    try:
        client.log_param(_current_run_id, key, str(value)[:500])
    except Exception as e:
        logger.warning(f"Failed to log param {key}: {e}")


def mlflow_set_tag(key: str, value: str) -> None:
    """
    Set a tag on the current run using MlflowClient.

    Args:
        key: Tag name
        value: Tag value
    """
    global _current_run_id

    if _current_run_id is None:
        return

    client = _get_mlflow_client()
    try:
        client.set_tag(_current_run_id, key, value)
    except Exception as e:
        logger.warning(f"Failed to set tag {key}: {e}")


def log_artifact(local_path: str, artifact_path: Optional[str] = None) -> None:
    """
    Log a local file or directory as an artifact using MlflowClient.

    Args:
        local_path: Path to local file or directory
        artifact_path: Optional destination path within artifacts
    """
    global _current_run_id

    if _current_run_id is None:
        return

    client = _get_mlflow_client()
    try:
        client.log_artifact(_current_run_id, local_path, artifact_path)
        logger.info(f"Logged artifact: {local_path}")
    except Exception as e:
        logger.warning(f"Failed to log artifact {local_path}: {e}")


def finish_mlflow(status: str = "FINISHED") -> None:
    """
    End the MLflow run using MlflowClient.

    Args:
        status: Run status - "FINISHED", "FAILED", or "KILLED"
    """
    global _current_run_id, _mlflow_client

    if _current_run_id is None:
        return

    client = _get_mlflow_client()

    # Map status string to MLflow RunStatus
    status_map = {
        "FINISHED": "FINISHED",
        "FAILED": "FAILED",
        "KILLED": "KILLED",
    }
    mlflow_status = status_map.get(status.upper(), "FINISHED")

    try:
        client.set_terminated(_current_run_id, status=mlflow_status)
        logger.info(f"MLflow run terminated: {_current_run_id} (status={mlflow_status})")
    except Exception as e:
        logger.warning(f"Failed to terminate run: {e}")

    # Reset global state
    _current_run_id = None


def get_current_run_id() -> Optional[str]:
    """Get the current MLflow run ID."""
    return _current_run_id


def get_mlflow_client() -> Optional["MlflowClient"]:
    """Get the current MlflowClient instance."""
    return _mlflow_client
```

### Step 2.2: Add MLflow Arguments

**Modify: `slime/utils/arguments.py`**

Add after the tensorboard arguments section (around line 1043):

```python
# MLflow logging arguments
def add_mlflow_arguments(parser):
    """Add MLflow-related arguments to parser."""
    group = parser.add_argument_group("MLflow")

    group.add_argument(
        "--use-mlflow",
        action="store_true",
        default=False,
        help="Enable MLflow logging.",
    )
    group.add_argument(
        "--mlflow-tracking-uri",
        type=str,
        default=None,
        help="MLflow tracking URI. Use 'databricks' for Databricks MLflow, "
             "or a path/URL for other backends. Defaults to MLFLOW_TRACKING_URI env var.",
    )
    group.add_argument(
        "--mlflow-experiment-name",
        type=str,
        default=None,
        help="MLflow experiment name. Required when using --use-mlflow.",
    )
    group.add_argument(
        "--mlflow-run-name",
        type=str,
        default=None,
        help="MLflow run name. If not specified, uses --mlflow-group with random suffix.",
    )
    group.add_argument(
        "--mlflow-group",
        type=str,
        default=None,
        help="Group tag for MLflow run, similar to --wandb-group.",
    )
    group.add_argument(
        "--mlflow-tags",
        type=json.loads,
        default=None,
        help='Additional tags as JSON dict, e.g., \'{"env": "prod"}\'',
    )
    group.add_argument(
        "--mlflow-run-id",
        type=str,
        default=None,
        help="Existing MLflow run ID to resume (internal use for distributed training).",
    )

    return parser
```

Then add to the parser chain in `get_slime_extra_args_provider`:

```python
def get_slime_extra_args_provider(add_custom_arguments=None):
    def add_slime_arguments(parser):
        # ... existing code ...
        parser = add_mlflow_arguments(parser)  # Add this line
        # ... rest of existing code ...
    return add_slime_arguments
```

### Step 2.3: Update Logging Utils

**Modify: `slime/utils/logging_utils.py`**

```python
"""Unified logging utilities for Slime."""

import logging

import wandb

from . import wandb_utils
from . import mlflow_utils  # NEW
from .tensorboard_utils import _TensorboardAdapter

logger = logging.getLogger(__name__)


def init_tracking(args, primary: bool = True, **kwargs):
    """
    Initialize all enabled tracking backends.

    Args:
        args: Parsed arguments with logging configuration
        primary: Whether this is the primary process (rank 0 in main controller)
        **kwargs: Additional kwargs passed to init functions
    """
    if primary:
        wandb_utils.init_wandb_primary(args, **kwargs)
        mlflow_utils.init_mlflow_primary(args, **kwargs)  # NEW
    else:
        wandb_utils.init_wandb_secondary(args, **kwargs)
        mlflow_utils.init_mlflow_secondary(args, **kwargs)  # NEW


def log(args, metrics, step_key: str):
    """
    Log metrics to all enabled backends.

    Args:
        args: Parsed arguments with logging configuration
        metrics: Dictionary of metrics to log (must include step_key)
        step_key: Key in metrics dict containing the step number (e.g., "train/step")
    """
    if args.use_wandb:
        wandb.log(metrics)

    if args.use_tensorboard:
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        _TensorboardAdapter(args).log(data=metrics_except_step, step=metrics[step_key])

    # NEW: MLflow logging
    if getattr(args, "use_mlflow", False):
        step = metrics.get(step_key, 0)
        metrics_except_step = {k: v for k, v in metrics.items() if k != step_key}
        mlflow_utils.mlflow_log_metrics(metrics_except_step, step=int(step))


def finish_tracking(args):
    """
    Finalize all tracking backends.

    Call this at the end of training for clean shutdown.
    """
    if getattr(args, "use_mlflow", False):
        mlflow_utils.finish_mlflow()

    # WandB and TensorBoard handle their own cleanup
    logger.info("Tracking finalized")
```

### Step 2.4: Add Cleanup Call to Train

**Modify: `train.py`**

Add after training completion (after `ray.get(rollout_manager.dispose.remote())`):

```python
# At the end of the train() function, before return:
from slime.utils.logging_utils import finish_tracking
finish_tracking(args)
```

---

## Phase 3: SPMD-style (torchrun) Launch Support

### Step 3.1: Create Launcher Module

**File: `slime/launcher/__init__.py`**

```python
"""SPMD-style launcher module for torchrun-based training."""

from .spmd_ray_bridge import (
    start_ray_server,
    init_ray_with_torch_distributed,
    get_node_ip,
    get_free_port,
)

__all__ = [
    "start_ray_server",
    "init_ray_with_torch_distributed",
    "get_node_ip",
    "get_free_port",
]
```

### Step 3.2: Implement SPMD-Ray Bridge

**File: `slime/launcher/spmd_ray_bridge.py`**

```python
"""
SPMD-style Ray cluster initialization using torch.distributed.

This module enables launching Slime training via torchrun (SPMD-style)
while still using Ray for actor management internally.

The pattern:
1. torchrun launches processes on all nodes with torch.distributed already set up
2. Rank 0 starts Ray head node
3. Each node's local rank 0 joins the Ray cluster
4. All processes wait for the full Ray cluster to be ready
5. Training proceeds as normal with Ray actors

Usage:
    with start_ray_server() as ray_address:
        if dist.get_rank() == 0:
            train(args)  # Only rank 0 runs the controller
"""

import logging
import os
import socket
import subprocess
import time
from contextlib import contextmanager
from typing import Generator

import ray
import torch.distributed as dist

logger = logging.getLogger(__name__)


def get_node_ip() -> str:
    """
    Get the IP address of the current node.

    Returns:
        str: The IP address, with any brackets removed
    """
    return ray.util.get_node_ip_address().strip("[]")


def get_free_port() -> int:
    """
    Get a free port number that can be used for binding a socket.

    Note: There is a small risk the port could be reclaimed by the system
    after this function returns and before the caller uses it.

    Returns:
        int: A free port number
    """
    with socket.socket() as sock:
        sock.bind(("", 0))
        return sock.getsockname()[1]


def is_cuda_visible_devices_set() -> bool:
    """
    Check if CUDA_VISIBLE_DEVICES is being managed by Ray.

    Returns:
        bool: True if Ray is setting CUDA_VISIBLE_DEVICES
    """
    return os.environ.get("RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES", "0") == "0"


def init_ray_with_torch_distributed(timeout_seconds: int = 300) -> str:
    """
    Initialize Ray cluster in a distributed PyTorch environment.

    This function sets up a Ray cluster where rank 0 starts the head node,
    and other nodes connect to it. It handles coordination between
    torch.distributed and Ray cluster initialization.

    Prerequisites:
        torch.distributed must already be initialized on all ranks
        (typically done by torchrun).

    Args:
        timeout_seconds: Maximum time to wait for all GPUs to be available

    Returns:
        str: The Ray cluster address (GCS address)

    Raises:
        RuntimeError: If the required GPUs are not available within timeout
        subprocess.CalledProcessError: If Ray start/stop commands fail
    """
    if not dist.is_initialized():
        raise RuntimeError(
            "torch.distributed must be initialized before calling "
            "init_ray_with_torch_distributed(). Use torchrun or call "
            "dist.init_process_group() first."
        )

    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))

    logger.info(f"Rank {rank}/{world_size} (local_rank={local_rank}) starting Ray init")

    if rank == 0:
        # Start Ray head on rank 0
        logger.info("Rank 0: Starting Ray head node")
        result = subprocess.run(
            ["ray", "start", "--head"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Failed to start Ray head: {result.stderr}")

        # Connect to the ray cluster
        ray.init("auto")

        # Get the GCS address for other nodes
        ctx = ray.get_runtime_context()
        address = ctx.gcs_address
        logger.info(f"Rank 0: Ray head started at {address}")
    else:
        address = ""

    # Broadcast address to all ranks
    address_list = [address]
    dist.broadcast_object_list(address_list, src=0)
    address = address_list[0]

    # Each node's local rank 0 joins the Ray cluster
    if rank != 0 and local_rank == 0:
        logger.info(f"Rank {rank}: Joining Ray cluster at {address}")
        result = subprocess.run(
            ["ray", "start", f"--address={address}"],
            capture_output=True,
            text=True,
            timeout=60,
        )
        if result.returncode != 0:
            raise RuntimeError(f"Rank {rank}: Failed to join Ray cluster: {result.stderr}")
        logger.info(f"Rank {rank}: Successfully joined Ray cluster")

    # Synchronize all ranks
    dist.barrier()

    # Wait for all GPUs to be available (rank 0 only)
    if rank == 0:
        start_time = time.time()
        while True:
            num_gpus = int(ray.cluster_resources().get("GPU", 0))
            if num_gpus >= world_size:
                break

            elapsed = time.time() - start_time
            if elapsed > timeout_seconds:
                raise RuntimeError(
                    f"Timeout after {timeout_seconds}s: Expected {world_size} GPUs, "
                    f"only {num_gpus} available."
                )

            logger.info(
                f"Waiting for GPUs: {num_gpus}/{world_size} "
                f"(elapsed: {elapsed:.1f}s, timeout: {timeout_seconds}s)"
            )
            time.sleep(5)

        logger.info(f"Ray cluster ready with {num_gpus} GPUs")
        logger.info(f"Available resources: {ray.available_resources()}")

    return address


@contextmanager
def start_ray_server(timeout_seconds: int = 300) -> Generator[str, None, None]:
    """
    Context manager for Ray server in a torch.distributed environment.

    This context manager handles the complete lifecycle of a Ray cluster:
    - Initializes torch.distributed if not already done
    - Starts the Ray cluster using init_ray_with_torch_distributed()
    - Provides the Ray address to the context
    - Ensures proper cleanup of Ray and distributed resources

    Yields:
        str: The Ray cluster address (GCS address)

    Example:
        >>> with start_ray_server() as ray_address:
        ...     if dist.get_rank() == 0:
        ...         # Run training controller
        ...         train(args)
        ... # Ray is automatically shut down here
    """
    init_torch_dist = False

    try:
        # Initialize torch.distributed if not already done
        if not dist.is_initialized():
            logger.info("Initializing torch.distributed with gloo backend")
            dist.init_process_group(backend="gloo")
            init_torch_dist = True

        # Initialize Ray cluster
        address = init_ray_with_torch_distributed(timeout_seconds)

        yield address

        # Keep all processes alive until training completes
        # This prevents resource reclamation before Ray operations finish
        dist.barrier()

    except Exception as e:
        logger.error(f"Error in start_ray_server: {e}")
        raise

    finally:
        # Cleanup in order: Ray first, then torch.distributed
        if dist.get_rank() == 0:
            try:
                logger.info("Rank 0: Shutting down Ray")
                ray.shutdown()
                subprocess.run(["ray", "stop", "--force"], check=False, timeout=30)
            except Exception as e:
                logger.warning(f"Error stopping Ray on rank 0: {e}")

        # Ensure all ranks wait for rank 0 cleanup
        dist.barrier()

        if init_torch_dist:
            logger.info(f"Rank {dist.get_rank()}: Destroying process group")
            dist.destroy_process_group()
```

### Step 3.3: Create SPMD Entry Point

**File: `slime/launcher/spmd_train.py`**

```python
"""
SPMD-style entry point for torchrun-based training.

This module provides an entry point for launching Slime training via torchrun,
enabling SPMD-style multi-node training.

Usage:
    # Single node with 4 GPUs
    torchrun --nproc-per-node=4 -m slime.launcher.spmd_train \\
        --config configs/training/qwen3-4b.yaml

    # Multi-node (run on each node)
    torchrun --nnodes=2 --nproc-per-node=4 \\
        --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \\
        -m slime.launcher.spmd_train \\
        --config configs/training/qwen3-4b.yaml
"""

import logging
import signal
import sys

import torch.distributed as dist

logger = logging.getLogger(__name__)


def setup_signal_handlers():
    """Setup handlers for graceful shutdown on SIGTERM/SIGINT."""
    def handler(signum, frame):
        logger.info(f"Received signal {signum}, initiating graceful shutdown")
        sys.exit(0)

    signal.signal(signal.SIGTERM, handler)
    signal.signal(signal.SIGINT, handler)


def spmd_train(args):
    """
    Entry point when launched via torchrun.

    This function:
    1. Sets up the Ray cluster using torch.distributed for coordination
    2. Runs the training controller on rank 0 only
    3. Keeps other ranks alive to maintain the Ray cluster

    Args:
        args: Parsed training arguments
    """
    from slime.launcher.spmd_ray_bridge import start_ray_server

    setup_signal_handlers()

    # Configure logging for distributed
    rank = dist.get_rank() if dist.is_initialized() else 0
    logging.basicConfig(
        level=logging.INFO,
        format=f"[Rank {rank}] %(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )

    logger.info(f"Starting SPMD training (rank={rank})")

    with start_ray_server(timeout_seconds=getattr(args, "spmd_ray_timeout", 300)) as ray_address:
        logger.info(f"Ray cluster ready at {ray_address}")

        if dist.get_rank() == 0:
            # Only rank 0 runs the controller logic
            logger.info("Rank 0: Running training controller")

            # Import and run the main train function
            # This is done inside the context to ensure Ray is ready
            from train import train
            train(args)

            logger.info("Rank 0: Training complete")
        else:
            # Other ranks just wait - they're needed to maintain the Ray cluster
            logger.info(f"Rank {dist.get_rank()}: Waiting for training to complete")

        # All ranks reach here when training is done
        logger.info(f"Rank {dist.get_rank()}: Exiting SPMD training")


def main():
    """Main entry point for command-line usage."""
    # Import parse_args here to avoid circular imports
    from slime.utils.arguments import parse_args

    args = parse_args()

    # Force launch_method to torchrun since we're running via this entry point
    args.launch_method = "torchrun"

    spmd_train(args)


if __name__ == "__main__":
    main()
```

### Step 3.4: Add Launch Method Arguments

**Modify: `slime/utils/arguments.py`**

Add to the cluster arguments section (around line 36-110):

```python
# Launch method arguments
parser.add_argument(
    "--launch-method",
    type=str,
    default="ray_job_submit",
    choices=["ray_job_submit", "torchrun"],
    help="Launch method: 'ray_job_submit' (default) uses Ray Job Submit, "
         "'torchrun' uses SPMD-style launch with torch.distributed.",
)
parser.add_argument(
    "--spmd-ray-timeout",
    type=int,
    default=300,
    help="Timeout in seconds for GPU availability when using torchrun launch. "
         "Only applies when --launch-method=torchrun.",
)
```

### Step 3.5: Update Main Entry Point

**Modify: `train.py`**

Update the `if __name__ == "__main__"` block:

```python
if __name__ == "__main__":
    args = parse_args()

    # Route to appropriate entry point based on launch method
    launch_method = getattr(args, "launch_method", "ray_job_submit")

    if launch_method == "torchrun":
        # Use SPMD-style launch
        from slime.launcher.spmd_train import spmd_train
        spmd_train(args)
    else:
        # Use existing Ray Job Submit launch
        train(args)
```

### Step 3.6: Add Guards for Pre-initialized Distributed

**Modify: `slime/ray/train_actor.py`**

Add guard in the `init()` method (around line 67):

```python
def init(self, args, role, with_ref=False):
    # ... existing setup code ...

    # Add guard for SPMD mode where torch.distributed is already initialized
    if not dist.is_initialized():
        dist.init_process_group(
            backend=backend,
            timeout=timedelta(minutes=args.distributed_timeout_minutes),
        )
    else:
        # In SPMD mode, get rank/world_size from existing group
        logger.info("torch.distributed already initialized (SPMD mode)")
        args.rank = dist.get_rank()
        args.world_size = dist.get_world_size()

    # ... rest of existing code ...
```

**Modify: `slime/backends/fsdp_utils/actor.py`**

Similar guard in `FSDPTrainRayActor.init()` (around line 67):

```python
# Add guard for SPMD mode
if not dist.is_initialized():
    dist.init_process_group(
        backend=backend,
        timeout=timedelta(minutes=args.distributed_timeout_minutes),
    )
```

### Step 3.7: Create Torchrun Launch Script

**File: `scripts/run-qwen3-4B-fsdp-torchrun.sh`**

```bash
#!/bin/bash
# SPMD-style torchrun launch for Slime training
#
# This script demonstrates launching Slime training using torchrun
# instead of Ray Job Submit. This approach is useful when:
# - Your cluster scheduler (e.g., SLURM) expects SPMD-style launches
# - You want to use torch.distributed for node coordination
# - You're integrating with existing torchrun-based infrastructure
#
# Usage:
#   Single node: ./run-qwen3-4B-fsdp-torchrun.sh
#   Multi-node:  See comments below

set -e

# Configuration
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-4}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}
CONFIG_FILE=${CONFIG_FILE:-"configs/training/qwen3-4b-grpo.yaml"}

# Optional: Override config values via CLI
EXTRA_ARGS=${EXTRA_ARGS:-""}

echo "=== SPMD Training Launch ==="
echo "Nodes: $NNODES"
echo "GPUs per node: $NPROC_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Config: $CONFIG_FILE"
echo "=========================="

# Single node launch
if [ "$NNODES" -eq 1 ]; then
    torchrun \
        --standalone \
        --nproc-per-node=$NPROC_PER_NODE \
        -m slime.launcher.spmd_train \
        --launch-method torchrun \
        --train-backend fsdp \
        --config "$CONFIG_FILE" \
        $EXTRA_ARGS
else
    # Multi-node launch (run this script on each node)
    # Requires MASTER_ADDR and MASTER_PORT to be set
    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m slime.launcher.spmd_train \
        --launch-method torchrun \
        --train-backend fsdp \
        --config "$CONFIG_FILE" \
        $EXTRA_ARGS
fi

echo "=== Training Complete ==="
```

Make it executable:
```bash
chmod +x scripts/run-qwen3-4B-fsdp-torchrun.sh
```

### Step 3.8: Create SPMD Runner with Compute YAML Support

This script enables launching training from a compute YAML configuration file (similar to MCLI/MosaicML format)
where the training parameters are embedded in a `parameters:` section.

**File: `slime/launcher/launch_grpo.py`**

```python
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


def extract_training_params(config: DictConfig) -> Dict[str, Any]:
    """
    Extract training-relevant parameters from config.

    Maps config keys to expected argument names.
    """
    # Direct mappings from config to args
    param_mapping = {
        "pretrain_model_name": "hf_checkpoint",
        "max_gen_len": "rollout_max_response_len",
        "max_model_len": "rollout_max_context_len",
        "generations_per_prompt": "n_samples_per_prompt",
        "global_train_batch_size": "global_batch_size",
        "max_duration": "num_rollout",
        "eval_interval": "eval_interval",
        "save_interval": "save_interval",
        "seq_parallel_world_size": "seq_parallel_world_size",
        "device_train_microbatch_size": "device_train_microbatch_size",
    }

    params = {}

    for config_key, arg_key in param_mapping.items():
        if config_key in config:
            params[arg_key] = config[config_key]

    return params


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

    if args.dry_run:
        logger.info("=== Dry Run - Config Summary ===")
        print(OmegaConf.to_yaml(config))
        return

    # Save merged config for reference
    merged_config_path = "/tmp/merged_training_config.yaml"
    OmegaConf.save(config, merged_config_path)
    logger.info(f"Saved merged config to: {merged_config_path}")

    # Extract MLflow settings if present
    use_mlflow = False
    mlflow_args = []
    if "loggers" in config and "mlflow" in config.loggers:
        mlflow_config = config.loggers.mlflow
        use_mlflow = True
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
        from slime.launcher.spmd_ray_bridge import start_ray_server
        from slime.utils.arguments import parse_args as slime_parse_args

        # Build args for slime
        slime_args_list = [
            "--launch-method", "torchrun",
            "--train-backend", args.train_backend,
            "--config", merged_config_path,
        ]
        slime_args_list.extend(mlflow_args)

        # Parse slime args
        sys.argv = ["train.py"] + slime_args_list
        training_args = slime_parse_args()

        # Run with SPMD launcher
        with start_ray_server() as ray_address:
            logger.info(f"Ray cluster ready at {ray_address}")

            if dist.get_rank() == 0:
                from train import train
                train(training_args)

            logger.info("Training complete")
    else:
        # Ray Job Submit mode - just parse args and call train
        from slime.utils.arguments import parse_args as slime_parse_args
        from train import train

        slime_args_list = [
            "--train-backend", args.train_backend,
            "--config", merged_config_path,
        ]
        slime_args_list.extend(mlflow_args)

        sys.argv = ["train.py"] + slime_args_list
        training_args = slime_parse_args()
        train(training_args)


if __name__ == "__main__":
    main()
```

### Step 3.9: Create Example Compute YAML Config

**File: `configs/compute/aroll-grpo-math.yaml`**

This example shows the full compute YAML format that works with the launcher:

```yaml
# Compute YAML for GRPO Math Training
# This format is compatible with MCLI/MosaicML cluster schedulers
#
# Usage:
#   python launch_grpo.py -f configs/compute/aroll-grpo-math.yaml
#   # Or with MCLI:
#   mcli run -f configs/compute/aroll-grpo-math.yaml

name: aroll-grpo-math-tool-workflow

image: mosaicml/dle:nightly-latest

scheduling:
  priority: high
  resumable: false
  preemptible: false
  max_duration: 24

compute:
  gpus: 16
  cluster: r5z2p2

integrations:
- integration_type: git_repo
  git_repo: your-org/slime
  ssh_clone: true
  git_branch: main

command: |-
  # Install dependencies
  pip install mlflow omegaconf

  # Launch GRPO training with SPMD
  cd /path/to/slime
  TORCH_COMPILE=1 python -m slime.launcher.launch_grpo \
    -f /mnt/config/parameters.yaml \
    --launch-method torchrun \
    --train-backend fsdp

# Training parameters (extracted by launch_grpo.py)
parameters:
  # Model settings
  pretrain_model_name: Qwen/Qwen3-4B-Thinking-2507
  max_gen_len: 8192
  max_model_len: 65536

  # Training settings
  generations_per_prompt: 8
  num_batches_per_update: 8
  global_train_batch_size: 64
  max_duration: 20iter
  eval_interval: 3iter
  save_interval: 100iter

  # Sequence parallelism
  seq_parallel_world_size: 4
  device_train_microbatch_size: 0.25

  # System prompt for tool use
  system_prompt: >-
    You are a helpful assistant that solves math problems.
    You have access to a Python execution tool to help verify calculations.
    To solve the complex math problems use minimal thinking in text format
    and check complex calculations via simple python execution (no plots or
    complicated libraries like sympy, numpy, etc.). You MUST use code as many
    times as possible to make sure your calculations and final answer is correct.

  # AROLL configuration
  aroll:
    env:
      name: base
      max_steps: 10
      tools:
      - name: python_exec
      prompt:
        system:
          prompt: ${system_prompt}
      rewards:
      - name: math_verifier
        params:
          verifier_score: 0.8
          format_score: 0.2
    strategy:
      name: n_gen
      n: ${generations_per_prompt}
    client:
      model: ${pretrain_model_name}
      api_key: nokey
      provider: orl-servers-vllm
      tool_call_parser: hermes
      reasoning_template: "\n<think>\n{reasoning_content}\n</think>\n\n{content}"

  # Mini-eval configuration
  minieval:
    concurrency:
      init_msgs: 2000
    evals:
    - name: math_competition
      wrappers:
      - name: agentic
        env_overrides:
          max_steps: 10
          tools:
          - name: python_exec
          prompt:
            system:
              prompt: ${system_prompt}

  # Model configuration
  model:
    converted_config_overrides:
      attn_config:
        seq_parallel_world_size: ${seq_parallel_world_size}

  # Tokenizer settings
  tokenizer:
    kwargs:
      pad_token: <|vision_pad|>

  # Data settings
  train_loader:
    dataset:
      remote: dbfs:/Volumes/datasets/wensun/data/deepscaler/aroll_mds/

  # Training variables
  variables:
    eos_token_ids:
    - 151643
    - 151645
    num_train_nodes: 1

  # Callbacks
  callbacks:
    hf_checkpointer:
      convert_to: qwen3

  # Loggers (MLflow)
  loggers:
    mlflow:
      tracking_uri: databricks
      experiment_name: test_orl_math_tool_aroll_grpo
      tags:
        group: grpo
```

### Step 3.10: Create SPMD Launch Wrapper Script

**File: `scripts/launch_spmd_training.sh`**

A convenience script that wraps `launch_grpo.py` with torchrun:

```bash
#!/bin/bash
# SPMD Training Launch Wrapper
#
# This script launches GRPO training using torchrun with a compute YAML config.
#
# Usage:
#   ./scripts/launch_spmd_training.sh -f configs/compute/aroll-grpo-math.yaml
#   ./scripts/launch_spmd_training.sh -f base.yaml -f override.yaml
#
# Environment variables:
#   NNODES          - Number of nodes (default: 1)
#   NPROC_PER_NODE  - GPUs per node (default: 8)
#   MASTER_ADDR     - Master node address (default: localhost)
#   MASTER_PORT     - Master node port (default: 29500)

set -e

# Configuration from environment
NNODES=${NNODES:-1}
NPROC_PER_NODE=${NPROC_PER_NODE:-8}
MASTER_ADDR=${MASTER_ADDR:-"localhost"}
MASTER_PORT=${MASTER_PORT:-29500}

# Pass all arguments to launch_grpo.py
CONFIG_ARGS="$@"

echo "=== SPMD Training Launch ==="
echo "Nodes: $NNODES"
echo "GPUs per node: $NPROC_PER_NODE"
echo "Master: $MASTER_ADDR:$MASTER_PORT"
echo "Config args: $CONFIG_ARGS"
echo "============================"

if [ "$NNODES" -eq 1 ]; then
    # Single-node launch
    torchrun \
        --standalone \
        --nproc-per-node=$NPROC_PER_NODE \
        -m slime.launcher.launch_grpo \
        --launch-method torchrun \
        $CONFIG_ARGS
else
    # Multi-node launch
    torchrun \
        --nnodes=$NNODES \
        --nproc-per-node=$NPROC_PER_NODE \
        --rdzv-backend=c10d \
        --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m slime.launcher.launch_grpo \
        --launch-method torchrun \
        $CONFIG_ARGS
fi

echo "=== Training Complete ==="
```

Make it executable:
```bash
chmod +x scripts/launch_spmd_training.sh
```

---

## Phase 4: Integration Testing

### Step 4.1: Test OmegaConf Config Loading

```bash
# Test config loading
python -c "
from slime.config.loader import load_config
config = load_config('configs/training/qwen3-4b-grpo.yaml')
print(f'Seed: {config.seed}')
print(f'Model: {config.pretrain_model_name}')
print(f'LR: {config.optimizer.lr}')
print(f'Max gen len: {config.max_gen_len}')
"

# Test variable interpolation
python -c "
from slime.config.loader import load_config
config = load_config('configs/training/qwen3-4b-grpo.yaml')
# max_seq_len should equal max_model_len due to interpolation
print(f'max_model_len: {config.max_model_len}')
print(f'rollout.max_context_len: {config.rollout.max_context_len}')
"

# Test CLI overrides
python -c "
from slime.config.loader import load_config
config = load_config(
    'configs/training/qwen3-4b-grpo.yaml',
    cli_overrides=['optimizer.lr=1e-7', 'seed=42']
)
print(f'LR (overridden): {config.optimizer.lr}')
print(f'Seed (overridden): {config.seed}')
"
```

### Step 4.2: Test MLflow Logging

```bash
# Test MLflow initialization (local tracking)
python -c "
import argparse
from slime.utils.mlflow_utils import init_mlflow_primary, mlflow_log_metrics, finish_mlflow

# Create mock args
args = argparse.Namespace(
    use_mlflow=True,
    mlflow_tracking_uri='./mlruns',
    mlflow_experiment_name='test_experiment',
    mlflow_run_name=None,
    mlflow_group='test_group',
    mlflow_tags=None,
)

# Initialize
init_mlflow_primary(args)
print(f'Run ID: {args.mlflow_run_id}')

# Log some metrics
mlflow_log_metrics({'train/loss': 0.5, 'train/accuracy': 0.8}, step=1)
mlflow_log_metrics({'train/loss': 0.3, 'train/accuracy': 0.9}, step=2)

# Finish
finish_mlflow()
print('MLflow test passed!')
"

# View logged runs
mlflow ui --backend-store-uri ./mlruns
```

### Step 4.3: Test Torchrun Launch (Single Node)

```bash
# Test with minimal config
torchrun --nproc-per-node=2 -m slime.launcher.spmd_train \
    --launch-method torchrun \
    --train-backend fsdp \
    --help

# Test actual launch (requires GPUs and model)
# NPROC_PER_NODE=2 ./scripts/run-qwen3-4B-fsdp-torchrun.sh
```

### Step 4.4: Test Combined Features

```bash
# Test config + MLflow + torchrun together
torchrun --nproc-per-node=2 -m slime.launcher.spmd_train \
    --config configs/training/qwen3-4b-grpo.yaml \
    --use-mlflow \
    --mlflow-experiment-name combined_test \
    --mlflow-tracking-uri ./mlruns
```

---

## File Summary

### New Files to Create

| File | Purpose |
|------|---------|
| `slime/config/__init__.py` | Config module exports |
| `slime/config/base.py` | Configuration dataclasses |
| `slime/config/loader.py` | OmegaConf YAML loader |
| `slime/config/converter.py` | Dataclass ↔ argparse converter |
| `slime/config/validation.py` | Configuration validation |
| `slime/utils/mlflow_utils.py` | MLflow logging utilities (using MlflowClient) |
| `slime/launcher/__init__.py` | Launcher module exports |
| `slime/launcher/spmd_ray_bridge.py` | SPMD-to-Ray coordination |
| `slime/launcher/spmd_train.py` | Torchrun entry point |
| `slime/launcher/launch_grpo.py` | GRPO launcher with compute YAML support |
| `configs/training/qwen3-4b-grpo.yaml` | Example training YAML config |
| `configs/compute/aroll-grpo-math.yaml` | Example compute YAML (MCLI format) |
| `scripts/run-qwen3-4B-fsdp-torchrun.sh` | Basic torchrun launch script |
| `scripts/launch_spmd_training.sh` | SPMD launch wrapper with config support |

### Existing Files to Modify

| File | Changes |
|------|---------|
| `slime/utils/arguments.py` | Add `--config`, `--launch-method`, MLflow args |
| `slime/utils/logging_utils.py` | Add MLflow dispatch, `finish_tracking()` |
| `train.py` | Add launch method routing, `finish_tracking()` |
| `slime/ray/train_actor.py` | Add `dist.is_initialized()` guard |
| `slime/backends/fsdp_utils/actor.py` | Add `dist.is_initialized()` guard |

---

## Rebase-Friendliness Summary

This implementation is designed to minimize merge conflicts during rebases:

1. **New directories** (`slime/config/`, `slime/launcher/`) - no conflicts
2. **New scripts** (`scripts/run-*-torchrun.sh`) - no conflicts
3. **Minimal changes to existing files** - only additive changes
4. **Backward compatible** - existing CLI args and Ray Job Submit continue working
5. **Configuration-driven** - new behavior controlled by new arguments

When rebasing:
- New directories will be cleanly added
- Changes to `arguments.py` are additions to the end of functions
- Changes to `train.py` are small additions to the main block
- Changes to actor files are guards that check existing state

---

## Quick Reference

### Using YAML Configs

```bash
# Basic usage
python train.py --config configs/training/qwen3-4b-grpo.yaml

# Override values via CLI
python train.py --config configs/training/qwen3-4b-grpo.yaml --lr 1e-7 --seed 42
```

### Using MLflow

```bash
# Local tracking
python train.py --use-mlflow --mlflow-experiment-name my_experiment --mlflow-tracking-uri ./mlruns

# Databricks tracking
python train.py --use-mlflow --mlflow-experiment-name my_experiment --mlflow-tracking-uri databricks
```

### Using Torchrun

```bash
# Single node
torchrun --nproc-per-node=4 -m slime.launcher.spmd_train --config configs/training/qwen3-4b-grpo.yaml

# Multi-node
torchrun --nnodes=2 --nproc-per-node=4 --rdzv-backend=c10d --rdzv-endpoint=$MASTER:29500 \
    -m slime.launcher.spmd_train --config configs/training/qwen3-4b-grpo.yaml
```

### Combined Usage

```bash
torchrun --nproc-per-node=4 -m slime.launcher.spmd_train \
    --config configs/training/qwen3-4b-grpo.yaml \
    --use-mlflow --mlflow-experiment-name my_run --mlflow-tracking-uri databricks
```

### Using Compute YAML (MCLI-style)

```bash
# Launch with compute YAML config
python -m slime.launcher.launch_grpo -f configs/compute/aroll-grpo-math.yaml

# Launch with multiple config files (merged in order)
python -m slime.launcher.launch_grpo -f base.yaml -f overrides.yaml

# Launch via wrapper script
./scripts/launch_spmd_training.sh -f configs/compute/aroll-grpo-math.yaml

# Multi-node launch with wrapper
NNODES=2 NPROC_PER_NODE=8 MASTER_ADDR=node0 ./scripts/launch_spmd_training.sh \
    -f configs/compute/aroll-grpo-math.yaml

# Dry run to see merged config
python -m slime.launcher.launch_grpo -f configs/compute/aroll-grpo-math.yaml --dry-run
```
