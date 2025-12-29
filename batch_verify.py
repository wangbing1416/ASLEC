import json
from math_verify import parse, verify
from tqdm import tqdm


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


def _verify(answer, think, gold):
    gold = gold.split('</think>')[-1]
    try:
        parsed_gold = parse(gold)
        parsed_think = parse(think)
        parsed_answer = parse(answer)
        flag1 = verify(gold, parsed_answer)
        flag1_1 = verify(parsed_gold, parsed_answer)
        flag2 = verify(gold, parsed_think)
        flag2_1 = verify(parsed_gold, parsed_think)
        answer_correct = True if flag1 or flag1_1 else False
        think_correct = True if flag2 or flag2_1 else False
    except:
        answer_correct, think_correct = False, False
    return answer_correct or think_correct

if __name__ == '__main__':
    # just some test cases, do not care it
    data = read_json('/testpath/open_qwen3_32b_10k_s5_32k/117.jsonl')
    print(data[0].keys())
    correct = _verify(answer=data[0]['distilled_answers'][0]['answer'],
                      think=data[0]['distilled_answers'][0]['think'], gold=data[0]['output'])
    print(correct)