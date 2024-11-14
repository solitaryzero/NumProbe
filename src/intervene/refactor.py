import os
import json
import argparse

tasks = [
    'Mistral-7B',
    'Mistral-7B_null',
    'Mistral-7B_random',
    'Mistral-7B_minus',
    'Mistral-7B_null_minus',
    'Mistral-7B_random_minus',
]


def process_file(args, file_name, greater=True):
    full_path = os.path.join(args.data_path, file_name)
    total, correct = 0, 0

    with open(full_path, 'r', encoding='utf-8') as fin:
        line = fin.readlines()[1]
        js = json.loads(line)
        for entry in js:
            total += 1
            clean_result, intervened_result = entry['clean_result'], entry['intervened_result']
            if (intervened_result <= 0) or (intervened_result > 100000):
                continue

            if (intervened_result > clean_result) == greater:
                correct += 1

    return correct/total


def refactor_results(
    args,
    param,
):
    results = {}
    for i in range(args.num_layers-param):
        r = process_file(args, 'res_layer_%d_to_%d.txt' %(i, i+param-1))
        results[i] = r

    return results


def main(args):
    for task in tasks:
        task_name = task
        results = refactor_results(
            args,
            param=6,
        )

        out_path = os.path.join(args.data_path, '%s_refactored.json' %task_name)
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/intervene/Mistral-7B')
    parser.add_argument('--num_layers', type=int, default=32)

    args = parser.parse_args()
    print(args)
    main(args)