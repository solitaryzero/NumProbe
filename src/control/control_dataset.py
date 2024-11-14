import json
import os
import random


if __name__ == '__main__':
    raw_path = './data/raw.jsonl'
    dataset = []
    with open(raw_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            num_digits = js['num_digits']
            control_signal = random.randint(10**(num_digits-1), (10**num_digits)-1)
            js['control_signal'] = control_signal

            dataset.append(js)

    out_path = './data/control.jsonl'
    with open(out_path, 'w', encoding='utf-8') as fout:
        for js in dataset:
            fout.write(json.dumps(js))
            fout.write('\n')