import json
import os
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import re

from transformers import AutoTokenizer
from sklearn.linear_model import LinearRegression


def read_jsonl(path):
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = []
        with open(path, "r", encoding='utf-8') as f:
            lines = f.readlines()
            for line in lines:
                data.append(json.loads(line))
    return data


def read_all_jsonl(path):
    all_data = []
    files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.jsonl')])
    for file in tqdm(files, desc=f"processing {path.split('/')[-1]}..."):
        data = read_jsonl(file)[0]
        all_data.append(data)

    print(f"Total read {len(all_data)} data entries")
    return all_data


def read_all_pkl(path):
    pkl_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pkl')])
    all_data = []
    # Traverse and read
    for file_path in tqdm(pkl_files, desc=f"processing {'/'.join(path.split('/')[-2:])}"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            all_data.append(data)
    return all_data

def write_jsonl(data, filename):
    """
    Write a list of dictionaries to a file, one JSON object per line.
    Args:
        data (list[dict]): List of dictionaries to write.
        filename (str): Output file path.
    """
    with open(filename, 'w', encoding='utf-8') as f:
        for item in data:
            # json.dumps converts dict to JSON string
            f.write(json.dumps(item, ensure_ascii=False) + '\n')


def split_step(text: str) -> list[str]:
    # pattern = r'(\. \n\n|\.\n\n|\? \n\n|\?\n\n)'
    pattern = r'(\. |\? )'
    parts = re.split(pattern, text)
    new_step_list = []
    for i in range(0, len(parts), 2):
        if not parts[i]:
            continue
        if i + 1 < len(parts) and parts[i+1]:
            combined_step = parts[i] + parts[i+1]
            new_step_list.append(combined_step)
        else:
            new_step_list.append(parts[i])
    return new_step_list


def output_drop_score(part_lengths, logprobs, skip_tokens=1):
    """
    Calculate average logprob, but skip the first skip_tokens tokens at the beginning of each sentence

    Args:
        part_lengths (list[int]): List of token lengths for each sentence
        logprobs (list[float]): Logprob list for corresponding tokens (in order)
        skip_tokens (int): Number of tokens to skip at the beginning of each sentence, default 1

    Returns:
        float: Average logprob after skipping specified tokens
    """
    filtered_logprobs = []
    index = 0

    for length in part_lengths:
        # Skip the first skip_tokens tokens of this sentence
        start = index + skip_tokens
        end = index + length
        # Note boundary protection, if skip_tokens >= length, skip the entire sentence
        if start < end:
            filtered_logprobs.extend(logprobs[start:end])
        index += length

    # Calculate average
    if filtered_logprobs:
        avg_logprob_drop = sum(filtered_logprobs) / len(filtered_logprobs)
    else:
        avg_logprob_drop = 0.0

    return avg_logprob_drop


def output_causal_score(part_lengths, logprobs, beta1=None, beta2=None, gamma=None, skip_tokens=1):
    """
    part_lengths: list[list[int]] List of step_lengths for each document
    logprobs: list[list[float]] List of logits for each document
    beta1, beta2, gamma: If provided use directly, otherwise fit with data
    skip_tokens: int, skip first few tokens of each step as "head" region, default=1
    """
    M_list = []
    M_first_list = []
    M_non_list = []
    F_list = []

    for logits, step_lengths in zip(logprobs, part_lengths):
        logits = np.array(logits)

        # Get indices of head tokens (skip according to skip_tokens count)
        first_indices = []
        idx = 0
        for sl in step_lengths:
            # For each sentence, head region is idx:(idx+skip_tokens)
            first_indices.extend(range(idx, min(idx + skip_tokens, idx + sl)))
            idx += sl

        first_indices = np.array(first_indices)
        non_indices = np.array([i for i in range(len(logits)) if i not in first_indices])

        def safe_mean(values):
            cleaned = [v for v in values if v is not None]
            return np.mean(cleaned) if cleaned else -16

        # Calculate means
        M_first = safe_mean([logits[i] for i in first_indices]) if len(first_indices) > 0 else -16
        M_non = safe_mean([logits[i] for i in non_indices]) if len(non_indices) > 0 else -16
        M = safe_mean(logits)

        # Head token ratio
        S = len(step_lengths)  # Number of steps
        T = len(logits)        # Number of tokens
        # Define F as "ratio of head tokens", can also keep previous sentence ratio
        F = len(first_indices) / T if T > 0 else 0

        M_list.append(M)
        M_first_list.append(M_first)
        M_non_list.append(M_non)
        F_list.append(F)

    M_arr = np.array(M_list)
    M_first_arr = np.array(M_first_list)
    M_non_arr = np.array(M_non_list)
    F_arr = np.array(F_list)

    if beta1 is None or beta2 is None or gamma is None:
        # Use linear regression to fit coefficients
        X = np.column_stack([M_non_arr, M_first_arr, F_arr])
        reg = LinearRegression().fit(X, M_arr)
        beta1, beta2, gamma = reg.coef_
        intercept = reg.intercept_
    else:
        intercept = 0.0

    # Bias removal processing
    M_adj = M_arr - gamma * F_arr

    return M_adj, beta1, beta2, gamma, intercept


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--skip_token", type=int, default=2)
    parser.add_argument("--teacher_model", type=str, default="ds_qwen32b", help="ds_qwen32b, qwen3_32b, qwq_32b")
    parser.add_argument("--student_model", type=str,
                        default="/yourpath/Qwen3-4B-Base-80k",
                        help="ds_qwen32b, qwen3_32b, qwq_32b")
    args = parser.parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.student_model)

    output_path = f'./limo/overall_limo-v2_{args.teacher_model}_s5_32k+ours_skip{args.skip_token}.jsonl'

    all_data = []
    part_length_list, logprobs_list = [], []
    suffix_list = ['', '_round2', '_round3', '_round4', '_round5']

    def tokenize(text):
        return tokenizer.convert_ids_to_tokens(tokenizer(text, add_special_tokens=False)["input_ids"])

    for suffix in suffix_list:
        data_path = f'./limo/limo-v2_{args.teacher_model}_s5_32k{suffix}'
        model_path = args.student_model
        logits_path = f'./limo/global_logprobs_limo-v2_{args.teacher_model}_s5_32k{suffix}'

        # load logits files
        local_path = f'./limo/local_logprobs_limo-v2_{args.teacher_model}_s5_32k{suffix}'
        # load logits files
        pkl_files = sorted([os.path.join(logits_path, f) for f in os.listdir(logits_path) if f.endswith('.pkl')])

        count = 0
        # Traverse and read
        for file_path in tqdm(pkl_files, desc=f"processing {'/'.join(logits_path.split('/')[-2:])}"):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)
                logits = data['logits']
                # assert len(logits) == len(local_logits)
                for s_i, solution in enumerate(logits):
                    try:
                        logit = solution['meta_info']['input_token_logprobs']
                        tokens = [tok[-1].replace('Ġ', ' ').replace('Ċ', '\n') for tok in
                                  solution['meta_info']['input_token_logprobs']]
                    except:
                        import ipdb; ipdb.set_trace()
                    output = "".join([tok[-1] for tok in logit])
                    logprobs = [tok[0] for tok in logit]

                    # First split by sentence
                    # sentences = re.split(pattern, output_text)
                    sentences = split_step(output)

                    pointer = 0  # Initial pointer position
                    step_start_token_ids = []

                    for sent in sentences:
                        if not sent.strip():
                            continue  # Skip empty sentence

                        # Tokenize the segment
                        seg_tokens = tokenize(' ' + sent)
                        seg_tokens = [tok.replace('Ġ', ' ').replace('Ċ', '\n') for tok in seg_tokens]
                        # seg_idx = tokenizer(sent, add_special_tokens=False)["input_ids"]

                        # Find the position of seg_tokens[0] in overall tokens
                        start_idx = None
                        # Set search range: 5 tokens before and after pointer
                        search_start = max(pointer - 3, 0)
                        search_end = min(pointer + 3, len(tokens))
                        # import ipdb; ipdb.set_trace()
                        for i in range(search_start, search_end):
                            if tokens[i].strip() == seg_tokens[0].strip():
                                start_idx = i
                                break

                        if start_idx is None:
                            # If not found, directly skip this solution
                            # import ipdb; ipdb.set_trace()
                            break

                        step_start_token_ids.append(start_idx)
                        pointer = start_idx + len(seg_tokens)  # Update pointer to end of current segment
                    # import ipdb; ipdb.set_trace()
                    if len(sentences) != len(step_start_token_ids):
                        continue
                    # Deduplicate and sort
                    step_start_token_ids = sorted(set(step_start_token_ids))
                    step_length = []
                    for i, start_idx in enumerate(step_start_token_ids):
                        end_idx = step_start_token_ids[i + 1] if i + 1 < len(step_start_token_ids) else len(tokens)
                        length = end_idx - start_idx
                        step_length.append(length)

                    part_lengths = step_length
                    # import ipdb; ipdb.set_trace()

                    if sum(part_lengths) != len(logit):
                        continue

                    # 0. ours drop variant
                    avg_logprob_drop = output_drop_score(part_lengths, logprobs, skip_tokens=args.skip_token)

                    # 0. ours causal variant
                    part_length_list.append(part_lengths)
                    logprobs_list.append(logprobs)

                    # 1. average log probability
                    avg_log_prob = np.mean([x for x in logprobs if x is not None])

                    parsed_data = {
                        'id': data['id'],
                        'teacher': args.teacher_model,
                        'input': data['input'],
                        'gt': data['output'],
                        'output': output,
                        'prompt_tokens': solution['meta_info']['prompt_tokens'],
                        'avg_log_prob': avg_log_prob,
                        'avg_logprob_drop': avg_logprob_drop
                    }
                    count += 1
                    all_data.append(parsed_data)
        print(f"there are {count} samples in {data_path.split('/')[-1]}")

    causal_output = output_causal_score(part_length_list, logprobs_list, skip_tokens=args.skip_token)
    print(f"fit parameters --> beta1 = {causal_output[1]}, beta1 = {causal_output[2]}, "
          f"gemma = {causal_output[3]}, intercept = {causal_output[4]}")
    causal_output = causal_output[0]

    with open(output_path, 'w', encoding='utf-8') as f:
        for item, avg_logprob_causal in zip(all_data, causal_output):
            # json.dumps converts dict to JSON string
            item['avg_logprob_causal'] = avg_logprob_causal
            f.write(json.dumps(item, ensure_ascii=False) + '\n')
    print(f"file has {len(all_data)} samples in {output_path}")



