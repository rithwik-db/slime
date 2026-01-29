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
