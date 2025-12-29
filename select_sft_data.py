import argparse
import json
import os
from tqdm import tqdm
from collections import defaultdict
import random


def read_jsonl(path):
    data = []
    with open(path, "r", encoding='utf-8') as f:
        lines = f.readlines()
        for line in tqdm(lines, desc=f"processing {path.split('/')[-1]}"):
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


def select_by_random(data, k):
    groups = group_by_field(data, field="id")
    selected_data = []
    for question in groups:
        selected_data.extend(random.sample(question, k))
    return selected_data

def select_by_max_global(data, k):
    groups = group_by_field(data, field="id")
    selected_data = []
    for question in groups:
        sorted_data = sorted(question, key=lambda x: x['avg_log_prob'], reverse=True)
        selected_data.extend(sorted_data[:k])
    return selected_data

def select_by_min_global(data, k):
    groups = group_by_field(data, field="id")
    selected_data = []
    for question in groups:
        sorted_data = sorted(question, key=lambda x: x['avg_log_prob'])
        selected_data.extend(sorted_data[:k])
    return selected_data

def select_by_single_teacher(data, teacher):
    groups = group_by_field(data, field="id")
    selected_data = []
    for question in groups:
        selected_data.extend([item for item in question if item.get('teacher') == teacher])
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
    parser.add_argument("--output_dir", type=str,
                        default="./sft_code/limo_s5_from_s20/limo_s5_max_global.jsonl")
    args = parser.parse_args()

    data = read_jsonl(args.input_path)
    data = select_by_max_global(data, args.top_k)
    # data = select_by_min_global(data, args.top_k)
    # data = select_by_random(data, args.top_k)
    # data = select_by_single_teacher(data, args.teacher_model)
    data = clean(data)
    write_list_to_jsonl(data, args.output_dir)
