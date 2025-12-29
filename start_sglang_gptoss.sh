#!/bin/bash

# environment variables
MASTER_ADDR="${LWS_LEADER_ADDRESS:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
RANK="${LWS_WORKER_INDEX:-0}"
NNODES="${LWS_GROUP_SIZE:-1}"
GPU=$(nvidia-smi -L | wc -l)

MODEL="${1:-/yourpath/gpt-oss-120b-bf16}"

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "NNODES: $NNODES"
echo "GPU: $GPU"
echo "MODEL: $MODEL"

if [ "$RANK" -eq 0 ]; then
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --dist-init-addr "${MASTER_ADDR}:5100" \
        --nnodes "${NNODES}" \
        --node-rank "${RANK}" \
        --host 127.0.0.1 \
        --port 30011 \
        --tp "${GPU}" \
        --trust-remote-code \
        --max-running-requests 256 \
        --mem-fraction-static 0.8 \
        --chunked-prefill-size 4096
else
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --dist-init-addr "${MASTER_ADDR}:5100" \
        --nnodes "${NNODES}" \
        --node-rank "${RANK}" \
        --host 127.0.0.1 \
        --port 30011 \
        --tp "${GPU}" \
        --trust-remote-code \
        --max-running-requests 256 \
        --mem-fraction-static 0.8 \
        --chunked-prefill-size 4096
fi
