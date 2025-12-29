import argparse
import os
from tqdm import tqdm
import json
from collections import defaultdict

def read_jsonl(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = []
        with open(path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f"reading {path.split('/')[-1]}"):
                data.append(json.loads(line))
    return data

def write_list_to_jsonl(data_list, file_path):
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in tqdm(data_list, desc=f"writing {file_path.split('/')[-1]}"):
            json_line = json.dumps(item, ensure_ascii=False)
            f.write(json_line + '\n')
    print(f"The file ({len(data_list)} items) has been written to {file_path}.")


def group_by_field(data, field="id"):
    groups = defaultdict(list)
    for item in data:
        groups[item[field]].append(item)
    return list(groups.values())

def clean_gptoss(item):
    a = "<|channel|>analysis<|message|>"
    b = "<|end|><|start|>assistant<|channel|>final<|message|>"

    output = item['output']

    # find positions of a and b
    idx_a = output.find(a)
    idx_b = output.find(b)

    if idx_a != -1 and idx_b != -1:
        between = output[idx_a + len(a): idx_b]
        after_b = output[idx_b + len(b):]
    else:
        between = ""
        after_b = ""
    item['output'] = f"<think> {between} </think> {after_b}"
    return item


def select_by_drop(data, k):
    groups = group_by_field(data, field="id")
    selected_data = []
    for question in groups:
        sorted_data = sorted(question, key=lambda x: x['avg_logprob_drop'], reverse=True)
        processed_top_k = []
        for item in sorted_data[:k]:
            if item.get('teacher') == 'gptoss':
                processed_top_k.append(clean_gptoss(item))
            else:
                processed_top_k.append(item)
        selected_data.extend(processed_top_k)
    return selected_data

def select_by_causal(data, k):
    groups = group_by_field(data, field="id")
    selected_data = []
    for question in groups:
        sorted_data = sorted(question, key=lambda x: x['avg_logprob_causal'], reverse=True)
        processed_top_k = []
        for item in sorted_data[:k]:
            if item.get('teacher') == 'gptoss':
                processed_top_k.append(clean_gptoss(item))
            else:
                processed_top_k.append(item)
        selected_data.extend(processed_top_k)
    return selected_data

def clean(data):
    return [{'id': item['id'], 'input': item['input'], 'output': item['output']} for item in data]

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--top_k", type=int, default=5)
    parser.add_argument("--teacher_model", type=str, default="gptoss")
    parser.add_argument("--student_model", type=str,
                        default="/yourpath/Qwen3-4B-Base-80k")
    parser.add_argument("--input_path", type=str,
                        default="./limo/overall_limo-v2_t4_s5_32k.jsonl")
    parser.add_argument("--output_drop_dir", type=str,
                        default="./sft_code/limo_s5_from_s20/limo_s5_drop_skip2.jsonl")
    parser.add_argument("--output_causal_dir", type=str,
                        default="./sft_code/limo_s5_from_s20/limo_s5_causal_skip2.jsonl")
    args = parser.parse_args()

    teacher_list = ['ds_qwen32b', 'qwen3_32b', 'qwq_32b', 'gptoss']
    selected_data = read_jsonl(args.input_path)
    # dict_keys(['id', 'teacher', 'input', 'gt', 'output', 'prompt_tokens', 'avg_log_prob', 'avg_entropy', 'ppl', 'boxed_output'])
    all_data = []
    for teacher in teacher_list:
        teacher_path = f'./limo/overall_limo-v2_{teacher}_s5_32k+ours_skip2.jsonl'
        # dict_keys(['id', 'teacher', 'input', 'gt', 'output', 'prompt_tokens', 'avg_log_prob', 'avg_local_log_prob', 'avg_entropy', 'ppl'])
        data = read_jsonl(teacher_path)
        teacher_specific_data = [record for record in selected_data if record.get("teacher") == teacher]
        # match avg_log_prob as key
        teacher_specific_data_keys = {item['avg_log_prob'] for item in teacher_specific_data}
        matched_list = [item for item in data if item['avg_log_prob'] in teacher_specific_data_keys]
        print(f"{teacher} model match {len(matched_list)} records")
        all_data += matched_list
    all_path = "./limo/overall_limo-v2_t4_s5_32k+ours_skip2.jsonl"
    write_list_to_jsonl(all_data, all_path)

    output_data = select_by_drop(all_data, args.top_k)
    output_data = clean(output_data)
    write_list_to_jsonl(output_data, args.output_drop_dir)

    output_data = select_by_causal(all_data, args.top_k)
    output_data = clean(output_data)
    write_list_to_jsonl(output_data, args.output_causal_dir)
