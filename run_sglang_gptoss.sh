#!/bin/bash

MODEL="/yourpath/gpt-oss-120b-bf16"

bash start_sglang_gptoss.sh "$MODEL" &
python gpu_stress.py 600
python data_syn_gptoss.py \
    --model "$MODEL" \
    --output_path ./limo/limo-v2_gptoss_s5_32k_round1 \
    --num_processes 150
