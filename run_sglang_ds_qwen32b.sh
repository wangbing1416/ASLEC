#!/bin/bash

MODEL="/yourpath/DeepSeek-R1-Distill-Qwen-32B"

bash start_sglang_ds_qwen32b.sh "$MODEL" &
python gpu_stress.py 300
python data_syn_qwen3_32b.py \
    --model "$MODEL" \
    --output_path ./limo/limo-v2_ds_qwen32b_s5_32k_round1 \
    --num_processes 128
