"""
SPMD-style entry point for torchrun-based training.

This module provides an entry point for launching Slime training via torchrun,
enabling SPMD-style multi-node training.

Usage:
    # Single node with 4 GPUs
    torchrun --nproc-per-node=4 -m slime.launcher.spmd_train \
        --config configs/training/qwen3-4b.yaml

    # Multi-node (run on each node)
    torchrun --nnodes=2 --nproc-per-node=4 \
        --rdzv-backend=c10d --rdzv-endpoint=$MASTER_ADDR:$MASTER_PORT \
        -m slime.launcher.spmd_train \
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
