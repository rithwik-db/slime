#!/bin/bash
# Rayless startup script using torchrun
# Run this same script on each node - runtime provides:
#   WORLD_SIZE, LOCAL_WORLD_SIZE, NODE_RANK, MASTER_ADDR, MASTER_PORT

set -ex

# Prevent buffering
export PYTHONUNBUFFERED=1

# Derive torchrun parameters from runtime-provided env vars
# WORLD_SIZE = total GPUs across all nodes
# LOCAL_WORLD_SIZE = GPUs per node
# NODE_RANK = this node's rank (0, 1, 2, ...)
NNODES=$((WORLD_SIZE / LOCAL_WORLD_SIZE))
NPROC_PER_NODE=${LOCAL_WORLD_SIZE}

echo "Node ${NODE_RANK}/${NNODES}: launching ${NPROC_PER_NODE} processes"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

# Launch with torchrun (sets RANK, LOCAL_RANK per process)
torchrun \
    --nnodes=${NNODES} \
    --nproc_per_node=${NPROC_PER_NODE} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    /root/slime/train.py \
    --use-rayless-init \
    --train-yaml /root/slime/examples/configs/qwen3-4B-grpo.yaml
