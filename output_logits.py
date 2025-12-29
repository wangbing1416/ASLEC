import json
import os
import pickle
import argparse
import requests
import sglang as sgl
from transformers import AutoTokenizer
from multiprocessing import Pool, Manager


def set_gpu_group(gpu_ids):
    """
    Set visible GPU group for current process.
    For example, gpu_ids = [0, 1, 2, 3]
    """
    os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(map(str, gpu_ids))
    print(f"Process {os.getpid()} has been assigned to GPU group: {os.environ['CUDA_VISIBLE_DEVICES']}")


def call_model_sglang(query, start_len, url):
    data = {
        "text": query,
        "logprob_start_len": start_len,
        "return_logprob": True,
        "return_text_in_logprobs": True,
        "top_logprobs_num": 5,
        "sampling_params": {"max_new_tokens": 0, "temperature": 0}
    }
    responses = requests.post(url, json=data, timeout=1200)
    return responses.json()


def read_one_line_jsonl(path):
    with open(path, 'r', encoding='utf-8') as f:
        data = json.load(f)
    return data


def write_pickle(result, output_file):
    with open(output_file, "wb") as f:
        pickle.dump(result, f)
    print(f"write results to {output_file.split('/')[-1]}")


def process_single_file(file, url, output_base_dir, tokenizer):
    id = file.split('.jsonl')[0].split('/')[-1]
    save_path = os.path.join(output_base_dir, f'{id}.pkl')
    if os.path.exists(save_path):
        print(f"Batch processing {id} already exists, skipping...")
        return id

    line = read_one_line_jsonl(file)
    problem, solution_list = line['input'], line['distilled_answers']

    # Construct base prompt
    prompt = f"<|im_start|>system\nYou are a helpful assistant.<|im_end|>\n" \
             f"<|im_start|>user\n{problem}<|im_end|>\n<|im_start|>assistant\n"
    length = len(tokenizer.tokenize(prompt))

    save_dict = {"id": id, "input": problem, "output": line['output']}

    # Construct batch queries & start_lens
    queries = []
    start_lens = []
    for solution in solution_list:
        if solution['think'] != '':
            think = f"<think>\n {solution['think']}  </think>\n {solution['answer']}"
        else:
            think = solution['answer']

        query = prompt + think
        tokens = tokenizer.encode(query)
        if len(tokens) > 80000:
            print(f"cut tokens to 80000 for {save_path} file")
            tokens = tokens[:80000]
            query = tokenizer.decode(tokens)
        queries.append(query)
        start_lens.append(length - 1)  # This is a fixed value, but can be adjusted as needed

    # One batch call
    try:
        resp = call_model_sglang(queries, start_lens, url)
        # results = resp.json()  # Batch results returned by server
    except Exception as e:
        print(f"Error processing batch {id}: {e}")
        # import traceback
        # traceback.print_exc()
        return None

    save_dict['logits'] = resp
    write_pickle(save_dict, save_path)
    return id


def main_hybrid_parallel(args, tokenizer, data_path, model_path, output_base_dir, url):
    input_filenames = [os.path.join(data_path, f) for f in os.listdir(data_path)
                       if f.endswith(".jsonl") and os.path.isfile(os.path.join(data_path, f))]

    tasks = [(file, url, output_base_dir, tokenizer) for file in input_filenames]

    with Pool(processes=args.num_processes) as pool:
        tasks = [
            pool.apply_async(
                process_single_file,
                args=task,
            )
            for task in tasks
        ]
        for task in tasks:
            task.get()  # Wait for all tasks to complete

    print("\nAll parallel tasks completed!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="ds_qwen32b", help="ds_qwen32b, qwen3_32b, qwq_32b, gptoss")
    parser.add_argument("--suffix", type=str, default="", help="_round2, _round3, _round4")
    parser.add_argument("--student_model", type=str,
                        default="/yourpath/Qwen3-4B-Base-80k", help="ds_qwen32b, qwen3_32b, qwq_32b")
    parser.add_argument("--num_processes", type=int, default=32)
    parser.add_argument("--url", type=str, default="http://127.0.0.1:30011/generate")
    args = parser.parse_args()

    data_path = f'./limo/limo-v2_{args.teacher_model}_s5_32k{args.suffix}'
    model_path = args.student_model
    output_path = f'./limo/global_logprobs_limo-v2_{args.teacher_model}_s5_32k{args.suffix}'
    os.makedirs(output_path, exist_ok=True)

    tokenizer = AutoTokenizer.from_pretrained(model_path)

    main_hybrid_parallel(args, tokenizer, data_path, model_path, output_path, args.url)



