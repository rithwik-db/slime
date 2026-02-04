#!/bin/bash
# Async rayless startup script - run ONE process per node
# Infrastructure must provide: NODE_RANK, WORLD_SIZE, LOCAL_WORLD_SIZE, MASTER_ADDR, MASTER_PORT
#
# Python reads these env vars directly and handles all orchestration logic.
# No env var manipulation needed in bash.

# Cleanup any existing processes
pkill -9 sglang 2>/dev/null || true
ray stop --force 2>/dev/null || true
sleep 3

set -ex
export PYTHONUNBUFFERED=1

echo "Node ${NODE_RANK}: Starting ASYNC rayless training"
echo "Infrastructure: WORLD_SIZE=${WORLD_SIZE}, LOCAL_WORLD_SIZE=${LOCAL_WORLD_SIZE}"
echo "Master: ${MASTER_ADDR}:${MASTER_PORT}"

# Python handles all orchestration logic - no env var manipulation needed
python /root/slime/train_async.py \
    --use-rayless-init \
    --train-yaml /root/slime/examples/configs/qwen3-4B-grpo.yaml
