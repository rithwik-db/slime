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
