# Jobs to be done:


- So this repo (Slime) uses ray to start training but my GPU cluster setup uses a SPMD (like torchrun) as input so previously I used the following code to be able to spin up a ray cluster by having all nodes run a file that does something like `_run_single_controller_ppo`, my goal is to figure out how to update this repo to support the same kind of launch process so that I can support multi-node training. How would I updated something like `run-qwen3-4B-fsdp.sh` to instead follow this kind of setup.

```python
# Copyright 2024 MosaicML ComposeRL authors
# SPDX-License-Identifier: Apache-2.0

import logging
import os
import socket
import subprocess
import time
from contextlib import contextmanager

import ray
import torch.distributed as dist

# Set up logger
logger = logging.getLogger(__name__)


def init_ray_with_torch_distributed(timeout_seconds: int = 30):
    """Initialize Ray cluster in a distributed PyTorch environment.

    This function sets up a Ray cluster where the master node (rank 0) starts the head node,
    and other nodes connect to it. It handles the coordination between PyTorch distributed
    training and Ray cluster initialization. It assumes torch.distributed
    is already initialized on all ranks and all the associated nodes are joining the ray cluster.

    The function:
    1. Starts Ray head node on rank 0
    2. Broadcasts the Ray address to all other ranks
    3. Connects worker nodes to the head node
    4. Waits for all GPUs to be available before proceeding

    Args:
        timeout_seconds (int): Maximum time to wait for GPUs to become available (default: 30)

    Returns:
        str: The Ray cluster address (GCS address) that can be used by other processes

    Raises:
        RuntimeError: If the required number of GPUs are not available within the timeout period
        subprocess.CalledProcessError: If Ray start/stop commands fail
    """
    # init ray on master node, rank 0
    if dist.get_rank() == 0:
        # Start Ray Server on master node
        subprocess.run(['ray', 'start', '--head'], check=True)
        # connect to the ray cluster
        ray.init('auto')
        # get existing ray ip and port
        ctx = ray.get_runtime_context()
        address = ctx.gcs_address
    else:
        address = ''
    address_list = [address]
    # broadcast address to all other ranks
    dist.broadcast_object_list(address_list, src=0)
    if dist.get_rank() != 0 and os.environ.get('LOCAL_RANK', None) == '0':
        address = address_list[0]
        logger.info(f'Rank {dist.get_rank()}: connecting to address {address}')
        subprocess.run(['ray', 'start', f'--address={address}'], check=True)
    dist.barrier()
    if dist.get_rank() == 0:
        # wait until num of gpus reach world_size
        num_gpus = int(ray.cluster_resources()['GPU'])
        start_time = time.time()
        while num_gpus < dist.get_world_size():
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise RuntimeError(
                    f'Timeout after {timeout_seconds}s: Failed to start {dist.get_world_size()} GPUs. Only {num_gpus} GPUs available.',
                )

            logger.info(
                f'Waiting for {dist.get_world_size() - num_gpus} GPUs to be available (elapsed: {elapsed_time:.1f}s, timeout: {timeout_seconds}s)',
            )
            num_gpus = int(ray.cluster_resources()['GPU'])
            # sleep ad-hoc 5s to avoid busy waiting
            time.sleep(5)

        logger.info(f'Total available GPUs: {ray.available_resources()}')
    return address


@contextmanager
def start_ray_server():
    """Context manager for Ray server in a torch distributed environment.

    This context manager handles the complete lifecycle of a Ray cluster:
    - Initializes PyTorch distributed process group if not already initialized
    - Starts the Ray cluster using init_ray_with_torch_distributed()
    - Provides the Ray address to the context
    - Ensures proper cleanup of Ray and distributed resources

    The context manager ensures that Ray is properly shut down and the distributed
    process group is destroyed even if an exception occurs.

    Yields:
        str: The Ray cluster address (GCS address)

    Example:
        >>> with start_ray_server() as ray_address:
        ...     # Use Ray cluster here
        ...     ray.get(some_remote_function.remote())
        ... # Ray is automatically shut down here
    """
    init_torch_dist = False
    if not dist.is_initialized():
        dist.init_process_group(backend='gloo')
        init_torch_dist = True
    address = init_ray_with_torch_distributed()
    try:
        yield address
        # NOTE we have to keep all the MCT orchestrator started processes alive with this barrier
        # until the ray cluster is stopped, otherwise the MCT orchestrator will reclaim the resources
        # once the processes on a node exit
        # this may time out too quick for a real world run, if so we might need to reuse the original
        # SyncActor based approach
        dist.barrier()
    finally:
        if dist.get_rank() == 0:
            ray.shutdown()
            subprocess.run(['ray', 'stop'], check=True)
        dist.barrier()
        if init_torch_dist:
            dist.destroy_process_group()


def get_node_ip():
    """Get the IP address of the current Ray node.

    Returns:
        str: The IP address of the current node, with any brackets removed

    Example:
        >>> ip = get_node_ip()
        >>> print(f"Current node IP: {ip}")
        Current node IP: 192.168.1.100
    """
    return ray.util.get_node_ip_address().strip('[]')


def get_free_port():
    """Get a free port number that can be used for binding a socket.

    This function creates a temporary socket, binds it to port 0 (which tells the OS
    to assign any available port), and returns the assigned port number. The socket
    is automatically closed when the context manager exits.

    NOTE there is a low risk that the port is recollected by the system after the context manager exits
    and before current process use it

    Returns:
        int: A free port number that can be used for network services

    Example:
        >>> port = get_free_port()
        >>> print(f"Available port: {port}")
        Available port: 54321
    """
    with socket.socket() as sock:
        sock.bind(('', 0))
        return sock.getsockname()[1]


def is_cuda_visible_devices_set():
    """Check if CUDA_VISIBLE_DEVICES environment variable is being set by Ray.

    Ray can automatically set the CUDA_VISIBLE_DEVICES environment variable to
    control which GPUs are visible to processes. This function checks whether
    this behavior is enabled or disabled.

    Returns:
        bool: True if Ray is setting CUDA_VISIBLE_DEVICES, False otherwise
    """
    return os.environ.get(
        'RAY_EXPERIMENTAL_NOSET_CUDA_VISIBLE_DEVICES',
        '0',
    ) == '0'
```
and
```python
def _run_single_controller_ppo(
    pretrain_model_path: str,
    world_size: int = 0,
):
    """Shared function for running single controller PPO.

    Args:
        pretrain_model_path: Path to the pretrained model
        world_size: Number of distributed processes
        prompts: List of prompts to test generation with
    """
    # Set vLLM attention backend to FLASH_ATTN otherwise FlashInfer backend
    # takes too long to jit compile
    os.environ['VLLM_ATTENTION_BACKEND'] = 'FLASH_ATTN'

    prompts = [
        'what is RAY?',
        'what is vLLM?',
    ]

    with start_ray_server() as _address:
        if dist.get_rank() == 0:
            # only rank 0 is the master controller

            # create SPMD training actors of the system
            if world_size == 0:
                world_size = dist.get_world_size()
            num_train_actors = world_size // 2
            train_actor = TrainActorGroup(num_train_actors)

            # Create vLLM engines (or inference actors)
            vllm_tensor_parallel_size = world_size - num_train_actors
            num_vllm_engines = (
                world_size - num_train_actors
            ) // vllm_tensor_parallel_size
            # TODO: Encapsulate this into a inference server manager class
            vllm_engines = create_vllm_engines(
                num_engines=num_vllm_engines,
                tensor_parallel_size=vllm_tensor_parallel_size,
                enforce_eager=True,
                pretrain=pretrain_model_path,
                revision=None,
                seed=1,
                enable_prefix_caching=False,
                max_model_len=512,
                device_bundle={
                    'GPU': 1,
                    'CPU': 1,
                    'worker_node': 0,
                },
            )
            inference_client = RolloutAgent(
                vllm_engines,
                vllm_tensor_parallel_size,
            )

            ppo_controller = PPOController(
                train_actor,
                inference_client,
                pretrain_model_path,
            )
            ppo_controller.train()

            inference_client.generate(prompts)
```

Secondly, I need to support mlflow logging instead of just tensorboard logging in this codebase, how would I add that.

Note that for these changes, I will be making a fork of the repo and a new branch and my goal is to keep the fork and the branch up to date with the main repo so I would need this to be implemented in a way that wouldn't break the script whenever we rebase.

Lastly, I want to be able to update the training configs so that I can use something like this

```yaml
parameters:
  seed: 17
  pretrain_model_name: deepseek-ai/DeepSeek-R1-Distill-Llama-8B
  max_gen_len: 8192
  max_model_len: 10240
  max_seq_len: ${max_model_len}
  generations_per_prompt: 8
  num_batches_per_update: 8
  global_train_batch_size: 64
  device_train_microbatch_size: 1  # Set to 1/spws when spws > 1
  seq_parallel_world_size: 1
  enable_packing: false
  rollout_method: unordered
  advantage_method: critic_free
  temperature: 1.0
  aroll:
    env:
      name: base
      max_steps: 1
      rewards:
      # TODO: The old yaml included bad_generation_end, short_response_reward
      # so we need to add them back in within the aroll code.
      - name: math_verifier
        params:
          verifier_score: 4.0
          format_score: 1.0
    strategy:
      name: n_gen
      n: ${generations_per_prompt}
    client:
      model: ${pretrain_model_name}
      api_key: nokey
      provider: orl-servers-vllm
    sampling:
      temperature: ${temperature}
      max_tokens: ${max_gen_len}
      presence_penalty: 0.0
      frequency_penalty: 0.0
    concurrency:
      tool: 1000
      # empirically scale reward workers to generations_per_prompt
      reward_workers: ${generations_per_prompt}
  minieval:
    evals:
    - name: gsm8k
      wrappers:
      - name: agentic
    - name: math_500
      wrappers:
      - name: agentic
    - name: math_hard
      wrappers:
      - name: agentic
    - name: math_competition
      wrappers:
      - name: agentic
  model:
    name: hf_critic_free_lm
    pretrained: true
    init_device: mixed
    use_auth_token: true
    attn_implementation: flash_attention_2
    allow_embedding_resizing: true
    use_flash_attention_2: true
    pretrained_model_name_or_path: ${pretrain_model_name}
    convert_to: mpt
    converted_config_overrides:
      attn_config:
        attn_uses_sequence_id: true
        seq_parallel_world_size: ${seq_parallel_world_size}
    algo_config:
      kl_config:
        estimator: k3
        clip_range: 40
        target_kl: 0.1
      iw_config:
        method: vanilla_is
        level: token
      policy_config:
        loss_type: grpo
        clip_ratio: 0.2
        length_normalize: true
      compute_ref_kl_loss: false
      advantage_normalization: standardized

  loggers:
    mlflow:
      tags:
        # Uncomment below to use a custom mlflow run name instead of the mcli run name
        # run: test_orl_aroll_grpo_math
        group: grpo
      tracking_uri: databricks
      experiment_name: test_orl_aroll_grpo_math
  callbacks:
    lr_monitor: {}
    scheduled_gc:
      batch_interval: 1000
    speed_monitor:
      window_size: 1
    memory_monitor: {}
    hf_checkpointer:
      overwrite: true
      precision: bfloat16
      save_folder: /tmp/hf_checkpoints/
      save_interval: 1dur
      convert_to: qwen2.5
    runtime_estimator: {}
  optimizer:
    lr: 1.0e-06
    name: decoupled_adamw
    betas:
    - 0.9
    - 0.95
    weight_decay: 0
  precision: amp_bf16
  scheduler:
    name: constant_with_warmup
    alpha: 1
    t_warmup: 0iter
    t_max: ${max_duration}
  tokenizer:
    name: ${pretrain_model_name}
    kwargs:
      padding: longest
      pad_token: <|finetune_right_pad_id|>
      truncation: true
      padding_side: left
      trust_remote_code: true
      model_max_length: ${max_model_len}
  variables:
    gamma: 1
    buffer:
      name: MinibatchRolloutBuffer
    lambda_gae: 1
    global_seed: ${seed}
    eos_token_ids:
    - 128001
    - 128008
    - 128009
    kl_controller:
      kl_ctl_type: fixed
      init_kl_coef: 0
    tokenizer_name: ${pretrain_model_name}
    reference_model:
      precision: amp_bf16
      pretrained: true
      model_config:
        name: hf_causal_lm
        pretrained: true
        use_auth_token: true
        use_flash_attention_2: true
        pretrained_model_name_or_path: ${pretrain_model_name}
        convert_to: mpt
        converted_config_overrides:
          attn_config:
            attn_uses_sequence_id: true
            seq_parallel_world_size: ${seq_parallel_world_size}
    generation_kwargs:
      top_p: 1
      do_sample: true
      use_cache: true
      temperature: ${temperature}
    epoch_per_iteration: 1
    generations_per_prompt: ${generations_per_prompt}
    num_batches_per_update: ${num_batches_per_update}
    device_generate_batch_size: 1
  # algorithms:
  #   gradient_clipping:
  #     clipping_type: norm
  #     clipping_threshold: 0.001
  autoresume: true
  log_config: true
  fsdp_config:
    sync_module_states: true
    verbose: false
    cpu_offload: false
    mixed_precision: PURE
    state_dict_type: sharded
    use_orig_params: true
    forward_prefetch: true
    backward_prefetch: BACKWARD_PRE
    sharding_strategy: FULL_SHARD
    activation_cpu_offload: false
    activation_checkpointing: true
    activation_checkpointing_reentrant: false
  save_folder: /tmp/checkpoints
  dist_timeout: 1800
  max_duration: 20iter
  progress_bar: false
  train_loader:
    name: aroll_init_msgs
    dataset:
      local: /tmp/train
      split: train
      remote: dbfs:/Volumes/kie/rithwikediga/dpsk_8b_open_r1_48k/with_cot/
      shuffle: true
      shuffle_seed: ${seed}
      download_timeout: 1800
    drop_last: true
    num_workers: 1
  enable_eval: true
  eval_interval: 3iter  # changed from 2iter to 3iter
  num_eval_servers: 0  # share with rollout servers
  save_interval: 100iter
  log_to_console: true
  save_overwrite: true
  python_log_level: info
  console_log_interval: 1ba
  device_eval_batch_size: 1
  eval_subset_num_batches: -1
  vllm_tensor_parallel_size: 1
  vllm_enable_prefix_caching: true
  vllm_monitor_server: true
  # num_train_gpus: 8 # This can be specified manually
  # num_vllm_gpus: 8 # This can be specified manually
  reset_prefix_cache: true
  enable_chunked_prefill: false
  partial_rollout: true
  save_num_checkpoints_to_keep: 1
  max_async_iter: 1
  trace_renewal_interval: 6
  train_policy: on-policy
```

And have the configs be populated through omegaconf yamls instead of bash scripts.