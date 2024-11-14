import os
import json
import argparse

tasks = {
    '14_start': ('start', 14),
    '5_layer': ('layer', 5),
    '6_layer': ('layer', 6),
}


def process_file(args, file_name):
    full_path = os.path.join(args.data_path, file_name)
    total, correct = 0, 0

    with open(full_path, 'r', encoding='utf-8') as fin:
        line = fin.readlines()[1]
        js = json.loads(line)
        for entry in js:
            total += 1
            clean_result, intervened_result = entry['clean_result'], entry['intervened_result']
            if (intervened_result == 0) or (intervened_result > 100000):
                continue

            if (intervened_result > clean_result):
                correct += 1

    return correct/total


def refactor_results(
    args,
    task_type,
    param,
):
    results = {}
    if (task_type == 'start'):
        for i in range(param, args.num_layers):
            r = process_file(args, 'res_layer_%d_to_%d.txt' %(param, i))
            results[i] = r
    elif (task_type == 'layer'):
        for i in range(args.num_layers-param):
            r = process_file(args, 'res_layer_%d_to_%d.txt' %(i, i+param-1))
            results[i] = r

    return results


def main(args):
    for task in tasks:
        task_name, task_type, p = task, tasks[task][0], tasks[task][1]
        results = refactor_results(
            args,
            task_type,
            p,
        )

        out_path = os.path.join(args.data_path, '%s_ref.json' %task_name)
        with open(out_path, 'w', encoding='utf-8') as fout:
            json.dump(results, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/intervene/Mistral-7B')
    parser.add_argument('--num_layers', type=int, default=32)

    args = parser.parse_args()
    print(args)
    main(args)