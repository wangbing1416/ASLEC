#!/bin/bash

# environment variables
MASTER_ADDR="${LWS_LEADER_ADDRESS:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
RANK="${LWS_WORKER_INDEX:-0}"
NNODES="${LWS_GROUP_SIZE:-1}"
GPU=$(nvidia-smi -L | wc -l)

MODEL="${1:-/yourpath/DeepSeek-R1-Distill-Qwen-32B}"

echo "MASTER_ADDR: $MASTER_ADDR"
echo "MASTER_PORT: $MASTER_PORT"
echo "RANK: $RANK"
echo "NNODES: $NNODES"
echo "GPU: $GPU"
echo "MODEL: $MODEL"

if [ "$RANK" -eq 0 ]; then
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --dist-init-addr "${MASTER_ADDR}:5000" \
        --nnodes "${NNODES}" \
        --node-rank "${RANK}" \
        --host 127.0.0.1 \
        --port 30011 \
        --reasoning-parser deepseek-r1 \
        --tp "${GPU}" \
        --trust-remote-code \
        --attention-backend flashinfer \
        --max-running-requests 512 \
        --mem-fraction-static 0.8 \
        --chunked-prefill-size 16384
        >> ./log/sglang_ds_qwen32b_port30011.log 2>&1
else
    python3 -m sglang.launch_server \
        --model-path "${MODEL}" \
        --dist-init-addr "${MASTER_ADDR}:5000" \
        --nnodes "${NNODES}" \
        --node-rank "${RANK}" \
        --host 127.0.0.1 \
        --port 30011 \
        --reasoning-parser deepseek-r1 \
        --tp "${GPU}" \
        --trust-remote-code \
        --attention-backend flashinfer \
        --max-running-requests 512 \
        --mem-fraction-static 0.8 \
        --chunked-prefill-size 16384
        >> ./log/sglang_ds_qwen32b_port30011.log 2>&1
fi
