#!/bin/bash

MODEL="/yourpath/QwQ-32B"

bash start_sglang_qwq32b.sh "$MODEL" &
python gpu_stress.py 300
python data_syn_qwq_32b.py \
    --model "$MODEL" \
    --output_path ./limo/limo-v2_qwq_32b_s5_32k_round1 \
    --num_processes 128
