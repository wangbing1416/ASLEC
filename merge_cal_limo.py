import json
import os
from tqdm import tqdm
import argparse
import pickle
import numpy as np
import math


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
    for file in tqdm(files, desc=f"processing {path.split("/")[-1]}..."):
        data = read_jsonl(file)[0]
        all_data.append(data)

    print(f"totally read {len(all_data)} data")
    return all_data


def read_all_pkl(path):
    pkl_files = sorted([os.path.join(path, f) for f in os.listdir(path) if f.endswith('.pkl')])
    all_data = []
    # reading
    for file_path in tqdm(pkl_files, desc=f"processing {'/'.join(path.split('/')[-2:])}"):
        with open(file_path, 'rb') as f:
            data = pickle.load(f)
            all_data.append(data)
    return all_data


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--teacher_model", type=str, default="ds_qwen32b", help="ds_qwen32b, qwen3_32b, qwq_32b")
    parser.add_argument("--student_model", type=str,
                        default="/yourpath/Qwen3-4B-Base-80k", help="ds_qwen32b, qwen3_32b, qwq_32b")
    args = parser.parse_args()

    output_path = f'./limo/overall_limo-v2_{args.teacher_model}_s5_32k+local.jsonl'
    with open(output_path, 'a', encoding='utf-8') as out_f:
        suffix_list = ['', '_round2', '_round3', '_round4', '_round5']
        for suffix in suffix_list:
            data_path = f'./limo/limo-v2_{args.teacher_model}_s5_32k{suffix}'
            model_path = args.student_model
            logits_path = f'./limo/global_logprobs_limo-v2_{args.teacher_model}_s5_32k{suffix}'

            # load logits files
            local_path = f'./limo/local_logprobs_limo-v2_{args.teacher_model}_s5_32k{suffix}'
            # load logits files
            pkl_files = sorted([os.path.join(logits_path, f) for f in os.listdir(logits_path) if f.endswith('.pkl')])
            local_pkl_files = sorted([os.path.join(local_path, f) for f in os.listdir(local_path) if f.endswith('.pkl')])
            # assert len(pkl_files) == len(local_pkl_files)
            all_data = []
            count = 0

            # reading
            for file_path in tqdm(pkl_files, desc=f"processing {'/'.join(logits_path.split('/')[-2:])}"):
                local_file_path = f'./limo/local_logprobs_limo-v2_{args.teacher_model}_s5_32k{suffix}/{file_path.split('/')[-1]}'
                # assert file_path.split('/')[-1] == local_file_path.split('/')[-1]
                if not os.path.exists(local_file_path):
                    print(f"the following file is not found: {local_file_path}")
                    continue
                with open(local_file_path, 'rb') as lcl_f:
                    local_data = pickle.load(lcl_f)
                    local_logits = local_data['logits']
                with open(file_path, 'rb') as f:
                    data = pickle.load(f)
                    logits = data['logits']
                    assert len(logits) == len(local_logits)
                    for s_i, solution in enumerate(logits):
                        try:
                            logit = solution['meta_info']['input_token_logprobs']
                        except:
                            import ipdb; ipdb.set_trace()
                        output = "".join([tok[-1] for tok in logit])
                        logprobs = [tok[0] for tok in logit if tok[0] is not None]
                        top_logprobs = []
                        for tok in solution['meta_info']['input_top_logprobs']:
                            if tok:
                                top_logprobs.append([top_tok[0] for top_tok in tok])
                        # 1. average log probability
                        avg_log_prob = np.mean(logprobs)
                        local_logprobs = local_logits[s_i]['mean_step_logp_list']
                        avg_local_logprob = sum(local_logprobs) / len(local_logprobs)
                        # 2. average entropy
                        entropies = []
                        for tok_top_logprobs in top_logprobs:
                            probs = np.exp(tok_top_logprobs)  # from log p to probability
                            probs /= probs.sum()  # normalize
                            entropy = -np.sum(probs * np.log(probs + 1e-12))
                            entropies.append(entropy)
                        avg_entropy = np.mean(entropies)
                        # 3. perplexity
                        ppl = math.exp(- avg_log_prob)

                        parsed_data = {
                            'id': data['id'],
                            'teacher': args.teacher_model,
                            'input': data['input'],
                            'gt': data['output'],
                            'output': output,
                            'prompt_tokens': solution['meta_info']['prompt_tokens'],
                            'avg_log_prob': avg_log_prob,
                            'avg_local_log_prob': avg_local_logprob,
                            'avg_entropy': avg_entropy,
                            'ppl': ppl,
                        }
                        count += 1
                        out_f.write(json.dumps(parsed_data, ensure_ascii=False) + '\n')
            print(f"there are {count} samples in {data_path.split('/')[-1]}")

