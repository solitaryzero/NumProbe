import random
import json
import os

tasks = {
    'main': {
        'sample_per_digit': 1000,
        'digit_counts': {d: 1000 for d in range(2, 11)},
        'file_name': 'raw.jsonl',
    },
    'extra': {
        'sample_per_digit': 10000,
        'digit_counts': {8: 10000},
        'file_name': '8.jsonl',
    },
    'extra2': {
        'sample_per_digit': 10000,
        'digit_counts': {4: 10000},
        'file_name': '4.jsonl',
    },
}

# task = 'main'
task = 'extra2'
sample_per_digit, digit_counts = tasks[task]['sample_per_digit'], tasks[task]['digit_counts']

data = {}

for d in digit_counts:
    rg = range(10**(d-1), 10**d)
    if (d == 2) or (d == 3) or (d == 4):
        prob_space = [(x, y) for x in rg for y in rg]
        samples = random.sample(prob_space, digit_counts[d])
        data[d] = samples
    else:
        xs = random.sample(rg, digit_counts[d])
        ys = random.sample(rg, digit_counts[d])
        samples = list(zip(xs, ys))
        data[d] = samples

out_path = './data'
if not(os.path.exists(out_path)):
    os.makedirs(out_path)

out_path = './data/%s' %(tasks[task]['file_name'])

cnt = 0
num_splits = [int(sample_per_digit*0.8), int(sample_per_digit*0.9), sample_per_digit]

with open(out_path, 'w', encoding='utf-8') as fout:
    for d in data:
        for i, sample in enumerate(data[d]):
            if (i < num_splits[0]):
                spl = 'train'
            elif (i < num_splits[1]):
                spl = 'val'
            else:
                spl = 'test'
            js = {
                'id': cnt,
                'num_digits': d,
                'a': sample[0],
                'b': sample[1],
                'golden': sample[0]+sample[1],
                'split': spl,
            }
            json.dump(js, fout)
            fout.write('\n')

            cnt += 1