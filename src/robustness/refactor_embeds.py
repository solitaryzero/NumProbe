import argparse
import os
import json
import numpy as np
import torch
from tqdm import tqdm


def refactor_prediction(record):
    # Does not use the LM prediction for robustness check
    record['predicted_answer'] = 1


def refactor_embeddings(record):
    all_layers = record['embeddings']
    num_layers = len(all_layers)
    record['num_layers'] = num_layers
    for i in range(num_layers):
        record['layer_%d' %i] = record['embeddings'][i].reshape((-1))


def annotate_correctness(record):
    record['label'] = (record['predicted_answer'] == record['golden'])


def main(args):
    splits = ['train', 'val', 'test']
    for spl in splits:
        data_path = os.path.join(args.data_path, '%s.data' %spl)
        with open(data_path, 'r', encoding='utf-8') as fin:
            all_records = [json.loads(line) for line in fin]
        for record in tqdm(all_records):
            refactor_prediction(record)
            annotate_correctness(record)

        if not(os.path.exists(args.out_path)):
            os.makedirs(args.out_path)
        out_path = os.path.join(args.out_path, '%s_processed.data' %spl)
        with open(out_path, 'w', encoding='utf-8') as fout:
            for record in all_records:
                json.dump(record, fout)
                fout.write('\n')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--out_path', type=str, default='./data/embeddings')
    args = parser.parse_args()

    print(args)
    main(args)