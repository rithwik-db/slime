"""
Utilities for initializing Ray cluster in an SPMD/rayless environment.

This module provides functions to initialize a Ray cluster using torch.distributed
for node coordination. It supports multiple environment variable conventions
commonly used by infrastructure schedulers (MCLI, Slurm, Kubernetes, torchrun).

Expected environment variables (set by infrastructure):
- NODE_RANK or RANK: Node rank (0, 1, 2, ...)
- NUM_NODES or NNODES or WORLD_SIZE/LOCAL_WORLD_SIZE: Total number of nodes
- LOCAL_WORLD_SIZE: Number of GPUs per node
- MASTER_ADDR: IP address of the master node
- MASTER_PORT: Port for torch.distributed rendezvous
"""

import logging
import os
import socket
import subprocess
import time
from contextlib import contextmanager

import ray
import torch
import torch.distributed as dist

# Set up logger
logger = logging.getLogger(__name__)


# =============================================================================
# Environment Variable Helpers
# =============================================================================


def get_node_rank() -> int:
    """Get this node's rank from infrastructure environment variables.

    Supports multiple conventions:
    - NODE_RANK: Explicit node rank (HPC schedulers, Slurm, MCLI)
    - RANK: When running one process per node, RANK is the node rank

    Returns:
        int: The node rank (0-indexed)
    """
    if "NODE_RANK" in os.environ:
        return int(os.environ["NODE_RANK"])
    return int(os.environ.get("RANK", 0))


def get_num_nodes() -> int:
    """Get total number of nodes from infrastructure environment variables.

    Supports multiple conventions:
    - NUM_NODES: Explicit node count
    - NNODES: torchrun convention
    - WORLD_SIZE/LOCAL_WORLD_SIZE: Compute from GPU count

    Returns:
        int: Total number of nodes
    """
    if "NUM_NODES" in os.environ:
        return int(os.environ["NUM_NODES"])
    if "NNODES" in os.environ:
        return int(os.environ["NNODES"])
    # Compute from GPU world size
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    local_world_size = int(os.environ.get("LOCAL_WORLD_SIZE", 1))
    return world_size // local_world_size


def get_local_world_size() -> int:
    """Get number of visible GPUs per node.

    Uses torch.cuda.device_count() as the source of truth since it
    respects CUDA_VISIBLE_DEVICES. Falls back to LOCAL_WORLD_SIZE
    if no CUDA devices are available (e.g., CPU-only mode).

    Returns:
        int: Number of visible GPUs per node
    """
    if "LOCAL_WORLD_SIZE" in os.environ:
        return int(os.environ["LOCAL_WORLD_SIZE"])
    return torch.cuda.device_count()


@contextmanager
def patch_env(**env_vars):
    """Context manager to temporarily set environment variables.

    Restores original values (or removes if not originally set) on exit.
    This prevents env var side effects from leaking to other code.

    Args:
        **env_vars: Environment variables to set temporarily.

    Example:
        >>> with patch_env(WORLD_SIZE="2", RANK="0"):
        ...     # env vars are set here
        ...     pass
        ... # env vars are restored here
    """
    original = {}
    for key, value in env_vars.items():
        original[key] = os.environ.get(key)
        os.environ[key] = str(value)

    try:
        yield
    finally:
        for key, orig_value in original.items():
            if orig_value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = orig_value


# =============================================================================
# Internal Helpers
# =============================================================================


def _setup_cuda_visible_devices():
    """Ensure CUDA_VISIBLE_DEVICES is set to all GPUs on the node.

    Returns the original value so it can be restored if needed.
    """
    original = os.environ.get("CUDA_VISIBLE_DEVICES")
    num_gpus = torch.cuda.device_count()
    if num_gpus > 0:
        all_gpus = ",".join(str(i) for i in range(num_gpus))
        os.environ["CUDA_VISIBLE_DEVICES"] = all_gpus
        logger.info(f"Set CUDA_VISIBLE_DEVICES={all_gpus}")
    return original


def init_ray_with_torch_distributed(timeout_seconds: int = 300):
    """Initialize Ray cluster using torch.distributed for node coordination.

    This function sets up a Ray cluster where node 0 (dist.get_rank() == 0) starts the head,
    and other nodes connect to it. It assumes:
    - torch.distributed is initialized with world_size = num_nodes (one process per node)
    - dist.get_rank() returns the NODE rank, not GPU rank
    - Each node has num_gpus_per_node GPUs

    The function:
    1. Starts Ray head node on node 0
    2. Broadcasts the Ray address to all other nodes via gloo
    3. Other nodes join the Ray cluster
    4. Waits for all GPUs to be available before proceeding
    """
    num_gpus_per_node = get_local_world_size()
    num_nodes = dist.get_world_size()
    expected_total_gpus = num_nodes * num_gpus_per_node

    logger.info(f"Node {dist.get_rank()}/{num_nodes}: num_gpus_per_node={num_gpus_per_node}, "
                f"expected_total_gpus={expected_total_gpus}")

    # Start Ray head on node 0
    if dist.get_rank() == 0:
        _setup_cuda_visible_devices()

        ray_start_cmd = [
            "ray", "start", "--head",
            f"--num-gpus={num_gpus_per_node}",
            "--disable-usage-stats",
        ]
        logger.info(f"Starting Ray head: {' '.join(ray_start_cmd)}")
        subprocess.run(ray_start_cmd, check=True)

        # Give Ray a moment to fully initialize
        time.sleep(3)

        # Connect to the ray cluster
        ray.init("auto")

        # Get Ray address
        address = ray.get_runtime_context().gcs_address
        logger.info(f"Ray head started at {address}")
    else:
        address = ""

    # Broadcast Ray address to all nodes via gloo
    address_list = [address]
    dist.broadcast_object_list(address_list, src=0)

    # Other nodes join the Ray cluster
    if dist.get_rank() != 0:
        address = address_list[0]
        logger.info(f"Node {dist.get_rank()}: connecting to Ray at {address}")

        # Ensure all GPUs are visible for Ray
        _setup_cuda_visible_devices()

        ray_start_cmd = [
            "ray", "start",
            f"--address={address}",
            f"--num-gpus={num_gpus_per_node}",
            "--disable-usage-stats",
        ]
        subprocess.run(ray_start_cmd, check=True)
        time.sleep(2)

    dist.barrier()

    # Wait until all GPUs are available
    if dist.get_rank() == 0:
        num_gpus = int(ray.cluster_resources().get("GPU", 0))
        start_time = time.time()
        while num_gpus < expected_total_gpus:
            elapsed_time = time.time() - start_time
            if elapsed_time > timeout_seconds:
                raise RuntimeError(
                    f"Timeout after {timeout_seconds}s: Expected {expected_total_gpus} GPUs but only {num_gpus} available.",
                )

            logger.info(
                f"Waiting for GPUs: {num_gpus}/{expected_total_gpus} available (elapsed: {elapsed_time:.1f}s)",
            )
            time.sleep(5)
            num_gpus = int(ray.cluster_resources().get("GPU", 0))

        logger.info(f"Ray cluster ready with {num_gpus} GPUs. Resources: {ray.available_resources()}")

    return address


@contextmanager
def start_ray_server(timeout_seconds: int = 300):
    """Context manager for Ray server in a torch distributed environment.

    This context manager handles the complete lifecycle of a Ray cluster:
    - Reads infrastructure env vars (NODE_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, etc.)
    - Sets up a gloo process group for node coordination
    - Starts the Ray cluster using init_ray_with_torch_distributed()
    - Ensures proper cleanup of Ray and distributed resources

    Yields:
        str: The Ray cluster address (GCS address)
    """
    should_manage_pg = not dist.is_initialized()

    num_nodes = get_num_nodes()
    node_rank = get_node_rank()
    master_addr = os.environ.get("MASTER_ADDR", "127.0.0.1")
    master_port = os.environ.get("MASTER_PORT", "29500")

    if should_manage_pg:
        logger.info(f"Setting up node sync: node {node_rank}/{num_nodes}, "
                    f"master={master_addr}:{master_port}")

    # We want to create a node level process group (each node is a rank in the process group)
    # since this allows us to run a command on node rank 0 to start the Ray cluster and then
    # all other nodes can join the Ray cluster by connecting to the Ray address.
    with patch_env(
        WORLD_SIZE=str(num_nodes),
        RANK=str(node_rank),
        MASTER_ADDR=master_addr,
        MASTER_PORT=str(master_port),
    ):
        if should_manage_pg:
            dist.init_process_group(backend="gloo")

        address = init_ray_with_torch_distributed(timeout_seconds=timeout_seconds)

        try:
            yield address
            # NOTE we have to keep all the MCT orchestrator started processes alive with this barrier
            # until the ray cluster is stopped, otherwise the MCT orchestrator will reclaim the resources
            # once the processes on a node exit
            dist.barrier()
        finally:
            if dist.get_rank() == 0:
                ray.shutdown()
                subprocess.run(["ray", "stop", "--force"], check=True)
            dist.barrier()
            if should_manage_pg:
                dist.destroy_process_group()
