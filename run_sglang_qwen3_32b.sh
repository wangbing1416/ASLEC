#!/bin/bash

MODEL="/yourpath/Qwen3-32B/"

bash start_sglang.sh "$MODEL" &
python gpu_stress.py 300
python data_syn_qwen3_32b.py \
    --model "$MODEL" \
    --output_path ./limo/limo-v2_qwen3_32b_s5_32k_round1 \
    --num_processes 128
