import random
import json

tasks = {
    'num_examples': 10000,
    'file_name': 'hard.jsonl',
}

n = tasks['num_examples']
data = []

for i in range(n):
    digit_x, digit_y = random.randint(2, 10), random.randint(2, 10)
    x = random.randint(10**(digit_x-1), 10**digit_x-1)
    y = random.randint(10**(digit_y-1), 10**digit_y-1)
    data.append((x, y))

out_path = './data/%s' %(tasks['file_name'])
cnt = 0
num_splits = [int(n*0.8), int(n*0.9), n]

with open(out_path, 'w', encoding='utf-8') as fout:
    for i, sample in enumerate(data):
        if (i < num_splits[0]):
            spl = 'train'
        elif (i < num_splits[1]):
            spl = 'val'
        else:
            spl = 'test'
        js = {
            'id': cnt,
            'num_digits': max(len(str(sample[0])), len(str(sample[1]))),
            'a': sample[0],
            'b': sample[1],
            'golden': sample[0]+sample[1],
            'split': spl,
        }
        json.dump(js, fout)
        fout.write('\n')

        cnt += 1