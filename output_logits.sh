#!/bin/bash

MODEL="/yourpath/Qwen3-4B-Base-80k"

bash start_sglang_qwen3_4b_base.sh "$MODEL" &   # start sglang
python gpu_stress.py 600     # waiting sglang starting
python 4_output_logits.py \
    --student_model "$MODEL" \
    --suffix _round1 \
    --teacher_model gptoss \
    --num_processes 32