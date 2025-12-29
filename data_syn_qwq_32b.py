import argparse
import os
import json
from tqdm import tqdm
from collections import defaultdict, OrderedDict
from typing import List, Dict, Any
from multiprocessing import Pool, Manager
import sglang as sgl
import openai
from batch_verify import _verify
import traceback


def call_model(query, client, model):
    response = client.chat.completions.create(
        model=model,
        messages=[
            {
            "role": "user",
            "content": query
            }
        ],
        stream=False,
        temperature=0.6,
        presence_penalty=0.0,
        frequency_penalty=0.0,
        top_p=0.95,
        max_tokens=32768,
        timeout=1200
    )
    return response.choices[0].message.content, response.choices[0].message.reasoning_content, response.usage.completion_tokens, response.usage.prompt_tokens


def read_json(path):
    data = []
    try:
        with open(path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in tqdm(lines, desc=f'processing {path.split("/")[-1]}'):
                data.append(json.loads(line))
    except:
        print(f'failed to read {path.split("/")[-1]}')
    return data

# write results to file
def write_results(result, output_file, chunk_i):
    with open(os.path.join(output_file, "%d.jsonl"%chunk_i), "w", encoding="utf-8") as f:
        for line in result:
            f.write(json.dumps(line, ensure_ascii=False) + "\n")


def group_by_field(data: List[Dict[str, Any]], field: str, keep_order: bool = True) -> List[List[Dict[str, Any]]]:
    grouped = OrderedDict() if keep_order else defaultdict(list)

    for item in data:
        if field not in item:
            raise KeyError(f"Field '{field}' not found in: {item}")
        key = item[field]

        if keep_order:
            # Initialize the group if the key is not seen before
            if key not in grouped:
                grouped[key] = []
            grouped[key].append(item)
        else:
            grouped[key].append(item)
    return list(grouped.values())


# check processed ids
def get_processed_ids(output_file):
    if not os.path.exists(output_file):
        return set(), list()
    res = set()
    existing_files = os.listdir(output_file)
    candidate_files = []
    for existing_file in existing_files:
        if existing_file.endswith('jsonl'):
            candidate_files.append(int(existing_file.split('.')[0]))
            with open(os.path.join(output_file, existing_file), "r", encoding="utf-8") as f:
                for line in f:
                    res.add(json.loads(line).get("id"))
    return res, candidate_files


# data processing
def process_data(data, client, model, sample_num=1):
    data_id = data['id']
    query = data['input']
    gold = data['output'].split('</think>')[-1]
    data['distilled_answers'] = []
    for _ in range(sample_num):
        try_time = 0
        wrong_time = 0
        while True:
            try:
                answer, think, output_token_length, input_token_length = call_model(query, client, model)
                # verify the generated answer
                if _verify(answer=answer, think=think, gold=gold):
                    data['distilled_answers'].append({"answer": answer, "think": think, "correct": True,
                                                      "usage": {"input_token_length": input_token_length,
                                                                "output_token_length": output_token_length}})
                    break
                else:
                    data['distilled_answers'].append({"answer": answer, "think": think, "correct": False,
                                                      "usage": {"input_token_length": input_token_length,
                                                                "output_token_length": output_token_length}})
                    wrong_time += 1

                if wrong_time >= 3:
                    print(f"data {data_id}'s wrong time >= 3, break")
                    break
            except Exception as e:
                print("[Exception type]:", type(e).__name__)
                traceback.print_exc()
                try_time = try_time + 1
                if try_time >= 3:
                    break
    return data


# multi processing data
def process_chunk(chunk, chunk_i, processed_ids, lock, ip, output_file, model, sample_num):
    if chunk_i in processed_ids:  # skip processed data
        pass
    else:
        client = openai.Client(base_url=ip, api_key="EMPTY")
        results = []
        for item in tqdm(chunk):
            result = process_data(item, client, model, sample_num)
            results.append(result)
        write_results(results, output_file, chunk_i)
        print(f"---->>> data {chunk_i}.jsonl written to {output_file}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--select_num', type=int, default=10000, help="number of selected samples")
    parser.add_argument('--chunk_size', type=int, default=1, help="chunk size")
    parser.add_argument('--num_processes', type=int, default=64, help="number of multiprocessing processes")
    parser.add_argument('--sample_num', type=int, default=5, help="number of samples per question")
    parser.add_argument('--data_path', type=str,
                        default='/yourpath/limo-v2+id.jsonl')
    parser.add_argument('--output_path', type=str, default='./limo-v2_qwen3_32b_s5_32k')
    parser.add_argument('--model', type=str, default='/yourpath/Qwen3-32B/')
    parser.add_argument('--ip', type=str, default='http://127.0.0.1:30011/v1')
    args = parser.parse_args()

    data = read_json(args.data_path)
    if not os.path.exists(args.output_path):
        os.makedirs(args.output_path)

    grouped = group_by_field(data, 'id')  # change input to id
    input_data = []
    for group in grouped:
        group[0]['num_solutions'] = len(group)
        input_data.append(group[0])
    input_data.sort(key=lambda x: x['id'])
    print(f'---->>> total {len(grouped)} inputs')

    processed_ids, processed_chunk_ids = get_processed_ids(args.output_path)
    print(f"---->>> processed_chunk_ids: {processed_chunk_ids}")

    chunks = [input_data[i:i + args.chunk_size] for i in range(0, len(input_data), args.chunk_size)]
    # import ipdb; ipdb.set_trace()
    with Manager() as manager:
        lock = manager.Lock()
        shared_processed_ids = manager.list(processed_chunk_ids)

        # start multi-processing pool
        with Pool(processes=args.num_processes) as pool:
            tasks = [
                pool.apply_async(
                    process_chunk,
                    args=(chunk, chunk_i, shared_processed_ids,
                          lock, args.ip, args.output_path, args.model, args.sample_num),
                )
                for chunk_i, chunk in enumerate(chunks)
            ]
            for task in tasks:
                task.get()  # all task completed
