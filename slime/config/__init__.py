"""Slime configuration system with Pydantic models and YAML support.

Note: Only FSDP backend is supported. Megatron backend has been disabled.
"""

from slime.config.models import (
    SlimeConfig,
    ClusterConfig,
    TrainConfig,
    RolloutConfig,
    DataConfig,
    AlgorithmConfig,
    EvaluationConfig,
    RewardConfig,
    WandbConfig,
    TensorboardConfig,
    DebugConfig,
    SGLangConfig,
    RouterConfig,
    FaultToleranceConfig,
    RolloutBufferConfig,
    MTPConfig,
    CIConfig,
    NetworkConfig,
)
from slime.config.loader import load_config, load_yaml_config, config_to_namespace

__all__ = [
    # Main config
    "SlimeConfig",
    # Section configs
    "ClusterConfig",
    "TrainConfig",
    "RolloutConfig",
    "DataConfig",
    "AlgorithmConfig",
    "EvaluationConfig",
    "RewardConfig",
    "WandbConfig",
    "TensorboardConfig",
    "DebugConfig",
    "SGLangConfig",
    "RouterConfig",
    "FaultToleranceConfig",
    "RolloutBufferConfig",
    "MTPConfig",
    "CIConfig",
    "NetworkConfig",
    # Loader functions
    "load_config",
    "load_yaml_config",
    "config_to_namespace",
]
