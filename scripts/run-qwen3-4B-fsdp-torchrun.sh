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
