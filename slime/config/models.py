"""
Pydantic configuration models for Slime training.

This file contains all configuration models in a single file for easy reading and review.
The models map to argument groups in slime/utils/arguments.py.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


# =============================================================================
# Base Configuration
# =============================================================================


class SlimeBaseConfig(BaseModel):
    """Base class for all Slime configuration models."""

    model_config = ConfigDict(
        extra="forbid",  # Fail on unknown fields to catch typos
        validate_assignment=True,  # Validate on attribute assignment
        use_enum_values=True,  # Use enum values instead of enum objects
        populate_by_name=True,  # Allow field aliases
    )


# =============================================================================
# Cluster Configuration (add_cluster_arguments)
# =============================================================================


class ClusterConfig(SlimeBaseConfig):
    """Cluster and resource allocation configuration."""

    actor_num_nodes: int = Field(default=1, description="Number of nodes for training actor")
    actor_num_gpus_per_node: int = Field(default=8, description="Number of GPUs per node for training actor")
    critic_num_nodes: int | None = Field(default=None, description="Number of nodes for critic model")
    critic_num_gpus_per_node: int | None = Field(default=None, description="Number of GPUs per node for critic")

    rollout_num_gpus: int | None = Field(
        default=None,
        description="Number of GPUs for inference. Ignored when using --colocate.",
    )
    rollout_num_gpus_per_engine: int = Field(
        default=1, description="Number of GPUs per inference engine (like tp_size in sglang)"
    )
    num_gpus_per_node: int = Field(default=8, description="Number of GPUs per node for rollout")

    colocate: bool = Field(default=False, description="Whether to colocate inference engines and actor")
    offload: bool = Field(default=False, description="Equivalent to --offload-train + --offload-rollout")
    offload_train: bool | None = Field(default=None, description="Whether to offload training actor to CPU")
    offload_rollout: bool | None = Field(default=None, description="Whether to offload rollout generator to CPU")

    distributed_backend: str = Field(default="nccl", description="Distributed backend")
    distributed_timeout_minutes: int = Field(default=10, description="Distributed timeout in minutes")


# =============================================================================
# Rayless Configuration (add_rayless_arguments)
# =============================================================================


class RaylessConfig(SlimeBaseConfig):
    """Configuration for rayless/SPMD-style startup (torchrun-compatible)."""

    use_rayless_init: bool = Field(
        default=False, description="Use rayless (torchrun/SPMD) initialization instead of Ray CLI"
    )
    ray_init_timeout: int = Field(default=300, description="Timeout in seconds for Ray cluster initialization")


# =============================================================================
# Training Configuration (add_train_arguments)
# =============================================================================


class TrainConfig(SlimeBaseConfig):
    """Training backend and format configuration."""

    train_backend: Literal["fsdp"] = Field(default="fsdp", description="Backend for training (only FSDP supported)")
    true_on_policy_mode: bool = Field(default=False, description="Whether to enable true-on-policy mode")
    train_env_vars: dict[str, Any] = Field(
        default_factory=dict, description="Extra environment variables for training process"
    )
    train_memory_margin_bytes: int = Field(
        default=1024**3, description="Margin for train memory allocation (default 1GB)"
    )
    enable_weights_backuper: bool = Field(default=True, description="Whether to enable weights backuper")
    log_probs_chunk_size: int = Field(default=-1, description="Chunk size to compute log probs to save memory")
    only_train_params_name_list: list[str] | None = Field(
        default=None, description="Regex patterns of parameter names to TRAIN (others frozen)"
    )
    freeze_params_name_list: list[str] | None = Field(
        default=None, description="Regex patterns of parameter names to FREEZE"
    )
    gradient_checkpointing: bool = Field(default=False, description="Enable gradient checkpointing to save memory")
    attn_implementation: str = Field(default="flash_attention_2", description="Attention implementation (flash_attention_2, flash_attention_3)")


# =============================================================================
# Rollout Configuration (add_rollout_arguments)
# =============================================================================


class RolloutConfig(SlimeBaseConfig):
    """Rollout generation configuration."""

    hf_checkpoint: str | None = Field(default=None, description="HuggingFace checkpoint path for the model")
    model_name: str | None = Field(default=None, description="Name of the model for weight conversion")
    rollout_function_path: str = Field(
        default="slime.rollout.sglang_rollout.generate_rollout", description="Path to rollout generation function"
    )

    # Sampling parameters
    rollout_temperature: float = Field(default=1.0, description="Temperature for inference during rollout")
    rollout_top_p: float = Field(default=1.0, description="Top-p for inference during rollout")
    rollout_top_k: int = Field(default=-1, description="Top-k for inference during rollout")

    # Context/length limits
    rollout_max_context_len: int | None = Field(default=None, description="Maximum context size for inference")
    rollout_max_prompt_len: int | None = Field(default=None, description="Maximum prompt length")
    rollout_max_response_len: int | None = Field(default=None, description="Maximum response length (max_tokens)")

    # Stop conditions
    rollout_skip_special_tokens: bool = Field(default=False, description="Skip special tokens in response")
    rollout_stop: list[str] | None = Field(default=None, description="Stop words for inference")
    rollout_stop_token_ids: list[int] | None = Field(default=None, description="Stop token IDs for inference")

    # Shuffle and seed
    rollout_shuffle: bool = Field(default=False, description="Whether to shuffle prompts during rollout")
    rollout_seed: int = Field(default=42, description="Seed for random number generator during rollout")

    # Sampling
    over_sampling_batch_size: int | None = Field(default=None, description="Granularity of sampling batch in rollout")
    dynamic_sampling_filter_path: str | None = Field(
        default=None, description="Filter function for dynamic sampling (e.g., DAPO)"
    )

    # Partial rollout
    partial_rollout: bool = Field(default=False, description="Whether to use partial rollout for long responses")
    mask_offpolicy_in_partial_rollout: bool = Field(
        default=False, description="Mask previous generation in partial rollout"
    )

    # Custom functions
    custom_generate_function_path: str | None = Field(
        default=None, description="Custom generate function for special rollout logic"
    )
    custom_rollout_log_function_path: str | None = Field(
        default=None, description="Custom function for logging rollout data"
    )
    custom_eval_rollout_log_function_path: str | None = Field(
        default=None, description="Custom function for logging eval rollout data"
    )
    buffer_filter_path: str | None = Field(default=None, description="Path to buffer filter function")

    # Weight update
    update_weight_buffer_size: int = Field(
        default=512 * 1024**2, description="Buffer size for update weight (in bytes)"
    )
    update_weights_interval: int = Field(default=1, description="Interval for updating weights")
    keep_old_actor: bool = Field(default=False, description="Whether to keep rollout model on training process")

    # Postprocessing
    rollout_data_postprocess_path: str | None = Field(
        default=None, description="Called after rollout data including log_probs"
    )

    # External rollout
    rollout_external: bool = Field(default=False, description="Use external SGLang instances")
    rollout_external_engine_addrs: list[str] | None = Field(
        default=None, description="Addresses of external engines"
    )


# =============================================================================
# Fault Tolerance Configuration (add_fault_tolerance_arguments)
# =============================================================================


class FaultToleranceConfig(SlimeBaseConfig):
    """Fault tolerance configuration for rollout."""

    use_fault_tolerance: bool = Field(default=False, description="Enable fault tolerance during rollout")
    rollout_health_check_interval: float = Field(
        default=30.0, description="Interval in seconds between health checks"
    )
    rollout_health_check_timeout: float = Field(
        default=30.0, description="Timeout for health check response"
    )
    rollout_health_check_first_wait: float = Field(
        default=0, description="Initial grace period before starting health checks"
    )


# =============================================================================
# Data Configuration (add_data_arguments)
# =============================================================================


class DataConfig(SlimeBaseConfig):
    """Dataset and batch configuration."""

    # Dataset iterations
    num_rollout: int | None = Field(default=None, description="Number of rollout steps")
    num_epoch: int | None = Field(default=None, description="Number of epochs (alternative to num_rollout)")

    # Dataset settings
    rollout_global_dataset: bool = Field(default=True, description="Use global dataset for rollout")
    data_source_path: str = Field(
        default="slime.rollout.data_source.RolloutDataSourceWithBuffer", description="Data source class path"
    )
    prompt_data: str | None = Field(default=None, description="Path to prompt data (JSONL)")

    # Chat template
    apply_chat_template: bool = Field(default=False, description="Apply chat template to input")
    apply_chat_template_kwargs: dict[str, Any] = Field(
        default_factory=dict, description="Kwargs for chat template"
    )

    # Data keys
    input_key: str = Field(default="input", description="JSON key for input/prompt")
    label_key: str | None = Field(default=None, description="JSON key for label/answer")
    metadata_key: str = Field(default="metadata", description="JSON key for metadata")
    tool_key: str = Field(default="tools", description="JSON key for tools")
    multimodal_keys: dict[str, str] | None = Field(default=None, description="Mapping of media types to data keys")

    # Start position
    start_rollout_id: int | None = Field(default=None, description="Starting rollout step")

    # Batch sizes
    rollout_batch_size: int | None = Field(default=None, description="Number of prompts per rollout step")
    n_samples_per_prompt: int = Field(default=1, description="Number of responses per prompt")
    global_batch_size: int | None = Field(default=None, description="Global batch size for training")
    num_steps_per_rollout: int | None = Field(default=None, description="Number of training steps per rollout")
    micro_batch_size: int = Field(default=1, description="Micro batch size for training")

    # Dynamic batching
    balance_data: bool = Field(default=False, description="Balance tokens between data parallel ranks")
    use_dynamic_batch_size: bool = Field(default=False, description="Use dynamic batch size based on tokens")
    max_tokens_per_gpu: int | None = Field(default=None, description="Maximum tokens per GPU for dynamic batching")
    log_probs_max_tokens_per_gpu: int | None = Field(
        default=None, description="Maximum tokens per GPU for log probs calculation"
    )


# =============================================================================
# Evaluation Configuration (add_eval_arguments)
# =============================================================================


class EvalDatasetEntry(SlimeBaseConfig):
    """Single evaluation dataset entry."""

    name: str = Field(description="Name of the evaluation dataset")
    path: str = Field(description="Path to the evaluation dataset")
    rm_type: str | None = Field(default=None, description="Reward model type for this dataset")
    input_key: str | None = Field(default=None, description="Override input key")
    label_key: str | None = Field(default=None, description="Override label key")
    tool_key: str | None = Field(default=None, description="Override tool key")
    metadata_key: str | None = Field(default=None, description="Override metadata key")
    n_samples_per_eval_prompt: int | None = Field(default=None, description="Override samples per prompt")
    temperature: float | None = Field(default=None, description="Override temperature")
    top_p: float | None = Field(default=None, description="Override top_p")
    top_k: int | None = Field(default=None, description="Override top_k")
    max_response_len: int | None = Field(default=None, description="Override max response length")
    stop: list[str] | None = Field(default=None, description="Override stop words")
    stop_token_ids: list[int] | None = Field(default=None, description="Override stop token IDs")
    min_new_tokens: int | None = Field(default=None, description="Minimum new tokens")
    custom_generate_function_path: str | None = Field(default=None, description="Custom generate function")
    metadata_overrides: dict[str, Any] = Field(default_factory=dict, description="Metadata overrides")


class EvaluationConfig(SlimeBaseConfig):
    """Evaluation configuration."""

    eval_function_path: str | None = Field(default=None, description="Path to eval generation function")
    eval_interval: int | None = Field(default=None, description="Evaluation interval in rollout steps")
    skip_eval_before_train: bool = Field(default=False, description="Skip evaluation before training")

    # Dataset specification (alternative to datasets list)
    eval_prompt_data: list[str] | None = Field(
        default=None, description="Eval prompt data as name/path pairs"
    )
    eval_config: str | None = Field(default=None, description="Path to OmegaConf eval config file")

    # Eval datasets (structured)
    datasets: list[EvalDatasetEntry] = Field(default_factory=list, description="List of evaluation datasets")

    # Default overrides for eval
    eval_input_key: str | None = Field(default=None, description="Override input key for eval")
    eval_label_key: str | None = Field(default=None, description="Override label key for eval")
    eval_tool_key: str | None = Field(default=None, description="Override tool key for eval")
    n_samples_per_eval_prompt: int = Field(default=1, description="Number of responses per eval prompt")
    eval_temperature: float | None = Field(default=None, description="Temperature for eval")
    eval_top_p: float | None = Field(default=None, description="Top-p for eval")
    eval_top_k: int | None = Field(default=None, description="Top-k for eval")
    eval_max_response_len: int | None = Field(default=None, description="Max response length for eval")
    eval_max_prompt_len: int | None = Field(default=None, description="Max prompt length for eval")
    eval_min_new_tokens: int | None = Field(default=None, description="Min new tokens for eval")
    eval_max_context_len: int | None = Field(default=None, description="Max context length for eval")


# =============================================================================
# Algorithm Configuration (add_algo_arguments)
# =============================================================================


class AlgorithmConfig(SlimeBaseConfig):
    """RL algorithm configuration (PPO/GRPO/REINFORCE++)."""

    # Checkpoints
    ref_load: str | None = Field(default=None, description="Checkpoint path for reference model")
    ref_ckpt_step: int | None = Field(default=None, description="Checkpoint step for reference model")
    load: str | None = Field(default=None, description="Checkpoint to load for training")
    save: str | None = Field(default=None, description="Path to save checkpoints")
    save_interval: int | None = Field(default=None, description="Checkpoint save interval")
    async_save: bool = Field(default=False, description="Enable async checkpoint saving")
    no_save_optim: bool = Field(default=False, description="Do not save optimizer state")
    save_hf: str | None = Field(default=None, description="Path to save HuggingFace format")

    # Training params
    seed: int = Field(default=1234, description="Random seed")
    clip_grad: float = Field(default=1.0, description="Gradient clipping value")
    calculate_per_token_loss: bool = Field(default=False, description="Calculate per-token loss")
    lr: float = Field(default=1e-6, description="Learning rate")

    # Optimizer settings (passed to FSDP backend)
    optimizer: str = Field(default="adam", description="Optimizer type (adam)")
    lr_decay_style: str = Field(default="constant", description="LR decay style (constant, linear, cosine)")
    weight_decay: float = Field(default=0.0, description="Weight decay")
    adam_beta1: float = Field(default=0.9, description="Adam beta1")
    adam_beta2: float = Field(default=0.95, description="Adam beta2")

    # Critic
    num_critic_only_steps: int = Field(default=0, description="Number of critic-only training steps")
    critic_load: str | None = Field(default=None, description="Checkpoint for critic model")
    critic_save: str | None = Field(default=None, description="Save path for critic model")
    critic_lr: float | None = Field(default=None, description="Learning rate for critic")
    critic_lr_warmup_iters: int = Field(default=0, description="Critic LR warmup iterations")

    # PPO clipping
    eps_clip: float = Field(default=0.2, description="PPO clip range")
    eps_clip_high: float | None = Field(default=None, description="PPO clip upper range")
    eps_clip_c: float | None = Field(default=None, description="Dual-clip PPO lower bound")
    value_clip: float = Field(default=0.2, description="Value loss clip")

    # KL settings
    kl_coef: float = Field(default=0.0, description="KL penalty coefficient for reward shaping")
    kl_loss_coef: float = Field(default=0.0, description="KL penalty coefficient for loss")
    use_kl_loss: bool = Field(default=False, description="Use KL loss from GRPO")
    kl_loss_type: Literal["k1", "k2", "k3", "low_var_kl"] = Field(default="k1", description="KL loss type")
    use_unbiased_kl: bool = Field(default=False, description="Enable unbiased KL estimation")
    ref_update_interval: int | None = Field(default=None, description="Interval to update ref model from actor")

    # Loss type
    loss_type: Literal["policy_loss", "sft_loss", "custom_loss"] = Field(default="policy_loss", description="Loss type")
    custom_loss_function_path: str | None = Field(default=None, description="Path to custom loss function")

    # Advantage estimation
    advantage_estimator: Literal[
        "grpo", "gspo", "reinforce_plus_plus", "reinforce_plus_plus_baseline", "ppo", "on_policy_distillation"
    ] = Field(default="grpo", description="Advantage estimator type")
    compute_advantages_and_returns: bool = Field(default=True, description="Compute advantages and returns")

    # PPO GAE
    gamma: float = Field(default=1.0, description="PPO GAE gamma")
    lambd: float = Field(default=1.0, description="PPO GAE lambda")

    # Normalization
    normalize_advantages: bool = Field(default=False, description="Normalize advantages")
    grpo_std_normalization: bool = Field(default=True, description="Dr.GRPO std normalization")
    rewards_normalization: bool = Field(default=True, description="Enable rewards normalization")

    # Entropy
    entropy_coef: float = Field(default=0.0, description="Entropy loss coefficient")
    use_rollout_entropy: bool = Field(default=False, description="Calculate entropy during logprobs computation")
    get_mismatch_metrics: bool = Field(default=False, description="Calculate mismatch metrics")
    reset_optimizer_states: bool = Field(default=False, description="Reset optimizer states after each rollout")

    # Importance sampling
    use_rollout_logprobs: bool = Field(default=False, description="Use rollout logprobs for importance sampling")
    use_tis: bool = Field(default=False, description="Enable TIS for off-policy importance sampling")
    tis_clip: float = Field(default=2.0, description="Clipping threshold for importance sampling")
    tis_clip_low: float = Field(default=0, description="Lower bound clipping for importance sampling")
    custom_tis_function_path: str | None = Field(default=None, description="Path to custom TIS function")
    custom_pg_loss_reducer_function_path: str | None = Field(
        default=None, description="Path to custom PG loss reducer"
    )

    # Routing replay
    use_routing_replay: bool = Field(default=False, description="Routing replay technique")
    use_rollout_routing_replay: bool = Field(default=False, description="Rollout routing replay technique")

    # OPSM
    use_opsm: bool = Field(default=False, description="Enable Off-Policy Sequence Masking")
    opsm_delta: float = Field(default=1e-4, description="Threshold for OPSM")

    @model_validator(mode="after")
    def validate_kl_settings(self) -> "AlgorithmConfig":
        if self.kl_coef != 0 and self.kl_loss_coef != 0:
            raise ValueError("Only one of kl_coef and kl_loss_coef can be set")
        return self

    @model_validator(mode="after")
    def validate_reinforce_advantages(self) -> "AlgorithmConfig":
        if self.advantage_estimator in ["reinforce_plus_plus", "reinforce_plus_plus_baseline"]:
            if not self.normalize_advantages:
                raise ValueError(
                    f"'{self.advantage_estimator}' requires normalize_advantages=True"
                )
        return self


# =============================================================================
# Router Configuration (add_router_arguments)
# =============================================================================


class RouterConfig(SlimeBaseConfig):
    """SlimeRouter configuration."""

    use_slime_router: bool = Field(default=False, description="Use SlimeRouter for text-based routing")
    slime_router_middleware_paths: list[str] = Field(default_factory=list, description="Middleware paths")
    slime_router_timeout: float | None = Field(default=None, description="Timeout for HTTP requests")
    slime_router_max_connections: int | None = Field(default=None, description="Max HTTP connections")
    slime_router_health_check_failure_threshold: int = Field(
        default=3, description="Failures before marking worker unhealthy"
    )
    # Note: Additional router args from RouterArgs.add_cli_args are passed through


# =============================================================================
# Wandb Configuration (add_wandb_arguments)
# =============================================================================


class WandbConfig(SlimeBaseConfig):
    """Weights & Biases logging configuration."""

    use_wandb: bool = Field(default=False, description="Enable W&B logging")
    wandb_mode: Literal["online", "offline", "disabled"] | None = Field(default=None, description="W&B mode")
    wandb_dir: str | None = Field(default=None, description="Directory for W&B logs")
    wandb_key: str | None = Field(default=None, description="W&B API key")
    wandb_host: str | None = Field(default=None, description="W&B host URL")
    wandb_team: str | None = Field(default=None, description="W&B team name")
    wandb_group: str | None = Field(default=None, description="W&B run group")
    wandb_project: str | None = Field(default=None, description="W&B project name")
    wandb_random_suffix: bool = Field(default=True, description="Add random suffix to run name")
    wandb_always_use_train_step: bool = Field(default=False, description="Always use train step as metric")
    wandb_run_id: str | None = Field(default=None, description="W&B run ID for resuming")
    log_multi_turn: bool = Field(default=False, description="Log multi-turn rollout info")
    log_passrate: bool = Field(default=False, description="Log pass@n metrics")
    log_reward_category: str | None = Field(default=None, description="Key for reward category logging")
    log_correct_samples: bool = Field(default=False, description="Log correct samples")


# =============================================================================
# Tensorboard Configuration (add_tensorboard_arguments)
# =============================================================================


class TensorboardConfig(SlimeBaseConfig):
    """TensorBoard logging configuration."""

    use_tensorboard: bool = Field(default=False, description="Enable TensorBoard logging")
    tb_project_name: str | None = Field(default=None, description="TensorBoard project name")
    tb_experiment_name: str | None = Field(default=None, description="TensorBoard experiment name")


# =============================================================================
# MLflow Configuration (add_mlflow_arguments)
# =============================================================================


class MlflowConfig(SlimeBaseConfig):
    """MLflow logging configuration."""

    use_mlflow: bool = Field(default=False, description="Enable MLflow logging")
    mlflow_tracking_uri: str | None = Field(
        default=None,
        description="MLflow tracking URI (defaults to 'databricks' if DATABRICKS_HOST env var is set)",
    )
    mlflow_experiment_name: str | None = Field(default=None, description="MLflow experiment name")
    mlflow_run_name: str | None = Field(default=None, description="MLflow run name")
    mlflow_run_id: str | None = Field(default=None, description="MLflow run ID for resuming")


# =============================================================================
# Debug Configuration (add_debug_arguments)
# =============================================================================


class DebugConfig(SlimeBaseConfig):
    """Debug and profiling configuration."""

    save_debug_rollout_data: str | None = Field(default=None, description="Path to save rollout data for debugging")
    load_debug_rollout_data: str | None = Field(default=None, description="Path to load rollout data for debugging")
    load_debug_rollout_data_subsample: float | None = Field(
        default=None, description="Subsample ratio of debug rollout data"
    )
    debug_rollout_only: bool = Field(default=False, description="Only run rollout generation without training")
    debug_train_only: bool = Field(default=False, description="Only run training without SGLang servers")
    save_debug_train_data: str | None = Field(default=None, description="Path to save train data for debugging")
    dump_details: str | None = Field(default=None, description="Dump all training details for analysis")
    memory_snapshot_dir: str = Field(default=".", description="Directory for memory snapshots")
    memory_snapshot_num_steps: int | None = Field(default=None, description="Number of steps for memory snapshot")
    profile_target: list[Literal["train_overall", "train_actor", "train_log_probs"]] = Field(
        default_factory=lambda: ["train_overall"], description="Profiling targets"
    )
    memory_recorder: Literal["torch", "memray"] = Field(default="torch", description="Memory recorder type")
    check_weight_update_equal: bool = Field(default=False, description="Check weight update equality")


# =============================================================================
# Network Configuration (add_network_arguments)
# =============================================================================


class NetworkConfig(SlimeBaseConfig):
    """Network configuration."""

    http_proxy: str | None = Field(default=None, description="HTTP proxy URL")
    use_distributed_post: bool = Field(default=False, description="Use distributed POST")
    http_timeout: float = Field(default=600.0, description="HTTP request timeout in seconds (default 10 minutes)")


# =============================================================================
# Reward Model Configuration (add_reward_model_arguments)
# =============================================================================


class RewardConfig(SlimeBaseConfig):
    """Reward model configuration."""

    rm_type: str | None = Field(default=None, description="Type of reward model")
    reward_key: str | None = Field(default=None, description="Key to extract reward from dict")
    eval_reward_key: str | None = Field(default=None, description="Key to extract eval reward from dict")
    group_rm: bool = Field(default=False, description="Apply RM on whole group")
    rm_url: str | None = Field(default=None, description="URL for remote reward model service")
    custom_rm_path: str | None = Field(default=None, description="Path to custom reward model function")
    custom_reward_post_process_path: str | None = Field(
        default=None, description="Path to custom reward post-processing function"
    )
    custom_convert_samples_to_train_data_path: str | None = Field(
        default=None, description="Path to custom samples-to-train-data converter"
    )


# =============================================================================
# Rollout Buffer Configuration (add_rollout_buffer_arguments)
# =============================================================================


class RolloutBufferConfig(SlimeBaseConfig):
    """Rollout buffer configuration."""

    rollout_buffer_url: str | None = Field(default=None, description="URL for rollout buffer")
    fetch_trajectory_retry_times: int = Field(default=-1, description="Retry times for fetching trajectory (-1=unlimited)")
    min_batch_collection_ratio: float = Field(default=1, description="Minimum batch collection ratio")
    rollout_task_type: str = Field(default="math", description="Rollout task type")
    loss_mask_type: Literal["qwen", "qwen3", "distill_qwen"] = Field(default="qwen", description="Loss mask type")
    data_pad_size_multiplier: int = Field(default=128, description="Multiplier for data padding size")
    rollout_sample_filter_path: str | None = Field(default=None, description="Path to rollout sample filter function")
    rollout_all_samples_process_path: str | None = Field(
        default=None, description="Path to process all samples including filtered ones"
    )
    disable_rollout_trim_samples: bool = Field(default=False, description="Disable trimming samples in buffer")
    use_dynamic_global_batch_size: bool = Field(default=False, description="Enable dynamic global batch size")


# =============================================================================
# MTP Training Configuration (add_mtp_training_arguments)
# =============================================================================


class MTPConfig(SlimeBaseConfig):
    """MTP (Multi-Token Prediction) training configuration."""

    mtp_num_layers: int | None = Field(default=None, description="Number of MTP layers")
    mtp_loss_scaling_factor: float = Field(default=0.2, description="MTP loss scaling factor")
    enable_mtp_training: bool = Field(default=False, description="Enable MTP layer parameter updates")


# =============================================================================
# Prefill/Decode Disaggregation Configuration
# =============================================================================


class PrefillDecodeConfig(SlimeBaseConfig):
    """Prefill/decode disaggregation configuration."""

    prefill_num_servers: int | None = Field(default=None, description="Number of prefill servers")


# =============================================================================
# CI Configuration (add_ci_arguments)
# =============================================================================


class CIConfig(SlimeBaseConfig):
    """CI testing configuration."""

    ci_test: bool = Field(default=False, description="Enable CI test mode")
    ci_disable_kl_checker: bool = Field(default=False, description="Disable KL checker in CI")
    ci_metric_checker_key: str | None = Field(default=None, description="Metric key for CI checker")
    ci_metric_checker_threshold: float | None = Field(default=None, description="Threshold for CI metric checker")
    ci_save_grad_norm: str | None = Field(default=None, description="Path to save gradient norms")
    ci_load_grad_norm: str | None = Field(default=None, description="Path to load gradient norms")


# =============================================================================
# SGLang Configuration (from add_sglang_arguments)
# =============================================================================


class SGLangConfig(SlimeBaseConfig):
    """SGLang inference engine configuration.

    Note: Most SGLang args are prefixed with 'sglang_' in the CLI.
    Additional args from sglang.ServerArgs are passed through.
    """

    model_config = ConfigDict(extra="allow")  # Allow SGLang passthrough args

    sglang_tensor_parallel_size: int | None = Field(default=None, description="Tensor parallel size for SGLang")
    sglang_mem_fraction_static: float | None = Field(default=None, description="Static memory fraction for SGLang")
    sglang_pp_size: int = Field(default=1, description="Pipeline parallel size for SGLang")
    sglang_pipeline_parallel_size: int = Field(default=1, description="Pipeline parallel size (alias)")
    sglang_decode_log_interval: int | None = Field(default=None, description="Decode log interval")
    sglang_chunked_prefill_size: int | None = Field(default=None, description="Chunked prefill size")
    sglang_attention_backend: str | None = Field(default=None, description="Attention backend (fa3, flashinfer)")


# =============================================================================
# Data Download Configuration
# =============================================================================


class DataDownloadEntry(SlimeBaseConfig):
    """Single data download entry for cloud storage."""

    source: str = Field(description="Cloud storage path (e.g., dbfs:/Volumes/catalog/schema/volume/file.jsonl)")
    destination: str = Field(description="Local filesystem path to save the file")


class DataDownloadConfig(SlimeBaseConfig):
    """Configuration for downloading datasets from cloud storage before training."""

    enabled: bool = Field(default=False, description="Enable cloud storage downloads")
    downloads: list[DataDownloadEntry] = Field(
        default_factory=list,
        description="List of files to download from cloud storage"
    )


# =============================================================================
# Root Configuration
# =============================================================================


class SlimeConfig(SlimeBaseConfig):
    """
    Root configuration for Slime training.

    This combines all section configs into a single validated configuration object.
    Extra fields at the root level are allowed for additional passthrough arguments.

    Note: Only FSDP backend is supported. Megatron backend has been disabled.
    """

    model_config = ConfigDict(extra="allow")  # Allow unknown args at root

    cluster: ClusterConfig = Field(default_factory=ClusterConfig)
    rayless: RaylessConfig = Field(default_factory=RaylessConfig)
    train: TrainConfig = Field(default_factory=TrainConfig)
    rollout: RolloutConfig = Field(default_factory=RolloutConfig)
    fault_tolerance: FaultToleranceConfig = Field(default_factory=FaultToleranceConfig)
    data: DataConfig = Field(default_factory=DataConfig)
    evaluation: EvaluationConfig = Field(default_factory=EvaluationConfig)
    algorithm: AlgorithmConfig = Field(default_factory=AlgorithmConfig)
    router: RouterConfig = Field(default_factory=RouterConfig)
    wandb: WandbConfig = Field(default_factory=WandbConfig)
    tensorboard: TensorboardConfig = Field(default_factory=TensorboardConfig)
    mlflow: MlflowConfig = Field(default_factory=MlflowConfig)
    debug: DebugConfig = Field(default_factory=DebugConfig)
    network: NetworkConfig = Field(default_factory=NetworkConfig)
    reward: RewardConfig = Field(default_factory=RewardConfig)
    rollout_buffer: RolloutBufferConfig = Field(default_factory=RolloutBufferConfig)
    mtp: MTPConfig = Field(default_factory=MTPConfig)
    prefill_decode: PrefillDecodeConfig = Field(default_factory=PrefillDecodeConfig)
    ci: CIConfig = Field(default_factory=CIConfig)
    sglang: SGLangConfig = Field(default_factory=SGLangConfig)
    data_download: DataDownloadConfig = Field(default_factory=DataDownloadConfig)

    def to_flat_dict(self) -> dict[str, Any]:
        """
        Convert the entire config to a flat dictionary for argparse compatibility.

        This flattens all nested configs into a single-level dict where keys
        match the argparse argument names (with underscores).
        """
        flat: dict[str, Any] = {}

        # Process each section
        for section_name in self.model_fields:
            section = getattr(self, section_name)
            if isinstance(section, SlimeBaseConfig):
                # Flatten section fields into root
                for field_name, value in section.model_dump().items():
                    if value is not None or field_name not in flat:
                        flat[field_name] = value

        # Add any extra fields at root level (passthrough for unmapped args)
        extra_fields = set(self.model_dump().keys()) - set(self.model_fields.keys())
        for field_name in extra_fields:
            flat[field_name] = getattr(self, field_name)

        return flat
