## ASLEC-DROP / ASLEC-CASL

The repo of our paper "_On the Step Length Confounding in LLM Reasoning Data Selection_"

---

**TL; DR**: We empirically find a step length collapse problem in previous naturalness-based LLM reasoning data selection, 
and solve it by proposing ASLEC-DROP / ASLEC-CASL two variant methods.

### Installing environment

You need two virtual environments to reproduce our methods.

1. **Data generation and output log probabilities by [SGLang (0.5.3rc0)](https://github.com/sgl-project/sglang).**

2. **SFT by [360-LLaMA-Factory](https://github.com/Qihoo360/360-LLaMA-Factory).**

You can follow their official installation guidance, 
and I recommend to directly use their official dockers.

### Data generation and SFT pipeline

We will release our generated data, if you access to these data, you can directly jump to _Step 3_.

**Step 1. Prepare data**

Downloading [LIMO-v2](https://huggingface.co/datasets/GAIR/LIMO-v2/blob/main/limo-v2.jsonl) 
and [AceReason-1.1-SFT](https://huggingface.co/datasets/nvidia/AceReason-1.1-SFT) (should be randomly sample 10k-20k questions and convert the file to .jsonl format). 

**Step 2. Generate responses**

First, deploy a locally LLM with SGLang on your own GPUs.

```shell
bash start_sglang_gptoss.sh
bash start_sglang_ds_qwen32b.sh
bash start_sglang_qwen3_32b.sh
bash start_sglang_qwq32b.sh
```

Then, run the data generation scripts.

```shell
python data_syn_qwen3_32b.py \
    --model /yourpath/DeepSeek-R1-Distill-Qwen-32B \
    --output_path ./limo/limo-v2_ds_qwen32b_s5_32k_round5 \
    --num_processes 128
```

or, we also provide one script to complete the above deployment and generation.

```shell
bash run_sglang_gptoss.sh
bash run_sglang_ds_qwen32b.sh
bash run_sglang_qwen3_32b.sh
bash run_sglang_qwq32b.sh
```

**Step 3. Output log probabilities**

Run the following script to output log probabilities.

```shell
bash output_logits.sh
```

**Step 4. Merge generated data**

Merge generated data and their log probabilities, 
and directly calculate metrics.

```shell
# calculate baseline metrics, e.g., GRACE, Local LP, ppl, entropy
python merge_cal_limo.py --teacher_model ds_qwen32b
# calculate our metrics
python merge_cal_limo_ours.py --teacher_model ds_qwen32b
```

**Step 5. Select data**

Select data by the calculated scores.

```shell
# select baseline data
python select_sft_data.py
# select our data
python select_sft_data_ours.py
```

**Step 6. SFT LLMs**

After activate `LlamaFactory` environment, run the following script to SFT the LLMs.
```shell
cd yourLlamaFactoryEnvPath
```
```shell
NNODES="${WORLD_SIZE:-1}"
MASTER_ADDR="${MASTER_ADDR:-localhost}"
MASTER_PORT="${MASTER_PORT:-29500}"
RANK="${RANK:-0}"
PROCESS_PER_NODE="${PROCESS_PER_NODE:-8}"
NGPUS="${TQ_GPU_NUM:-8}"

EXPERIMENT_CONFIG="/rootpath/sft_code/src/full_sft_qwen3_4B_base_acereason_s1_random.yaml"

torchrun --nproc_per_node $NGPUS --nnodes $NNODES \
    --rdzv_id=3649 --rdzv_backend=c10d  --rdzv_endpoint=${MASTER_ADDR}:${MASTER_PORT} \
    src/llamafactory/launcher.py $EXPERIMENT_CONFIG
```

**Step 6. Evaluation**

Following deepscaler

### Citation
```

```