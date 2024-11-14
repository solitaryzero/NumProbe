import os
import argparse
import torch
import json
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model


def read_dataset(args):
    control_map = {}
    with open(args.control_path, 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            eid, control = js['id'], js['control_signal']
            control_map[eid] = control


    splits = ['train', 'val', 'test']
    processed_records = {}
    num_examples = {}
    for spl in splits:
        full_path = os.path.join(args.data_path, '%s_processed.data' %spl)
        with open(full_path, 'r', encoding='utf-8') as fin:
            all_records = [json.loads(line) for line in fin]

        data_dict = {}
        for record in all_records:
            for key in record:
                if (key not in data_dict):
                    data_dict[key] = []
                data_dict[key].append(record[key])

            signal_name = 'control_signal'
            if (signal_name not in data_dict):
                data_dict[signal_name] = []
            data_dict[signal_name].append(control_map[record['id']])

        num_examples[spl] = len(data_dict['id'])
        processed_records[spl] = pd.DataFrame.from_dict(data_dict)

    return processed_records, num_examples


def read_embeds(args, num_examples, split, layer):
    full_path = os.path.join(args.data_path, '%s_layer_%d.embeds' %(split, layer+1)) # ignore first raw embedding layer
    embeddings = []
    with open(full_path, 'rb') as fin:
        for i in range(num_examples):
            embeddings.append(np.load(fin))

    return embeddings


def model_factory(args):
    model_map = {
        'none': linear_model.LinearRegression,
        'lasso': linear_model.Lasso,
        'ridge': linear_model.Ridge,
    }

    if (args.penalty == 'none'):
        return model_map[args.penalty]()
    else:
        return model_map[args.penalty](alpha=args.alpha)


def main(args):
    full_dataset, num_examples = read_dataset(args)

    # training
    train_dataset = full_dataset['train']
    num_train_examples = num_examples['train']
    num_layers = args.num_layers

    linears = {
        'control_signal': [],
    }

    for i in tqdm(range(num_layers)): # ignore raw embedding layer
        for target in linears:
            Y = train_dataset[target]

            all_X = read_embeds(args, num_train_examples, 'train', i)
            selected_X = []
            for j, X in enumerate(all_X):
                special_positions = train_dataset['special_positions'][j]
                num1_start, num1_end, num2_start, num2_end, last_token = special_positions
                key_indices = list(range(num1_end, num2_start)) \
                            + list(range(num2_end, last_token+1))
                selected_X.append(X[key_indices, :])

            selected_X = np.stack(selected_X, axis=0)
            layer_regressors = []
            for j in range(selected_X.shape[1]):
                Y_train = Y
                X_train = selected_X[:, j, :]

                if (args.logscale_prediction):
                    Y_train = np.log2(Y_train)
                else:
                    Y_train = Y_train
                regr = model_factory(args)
                regr.fit(X_train, Y_train)

                layer_regressors.append(regr)
            
            linears[target].append(layer_regressors)

    # score on validation set
    valid_dataset = full_dataset['val']
    num_valid_examples = num_examples['val']

    # calculate score on the valid set
    scores = {
        'control_signal': [],
    }

    for i in tqdm(range(num_layers)): # ignore raw embedding layer
        for target in linears:
            Y = valid_dataset[target]

            all_X = read_embeds(args, num_valid_examples, 'val', i)
            selected_X = []
            for j, X in enumerate(all_X):
                special_positions = valid_dataset['special_positions'][j]
                num1_start, num1_end, num2_start, num2_end, last_token = special_positions
                key_indices = list(range(num1_end, num2_start)) \
                            + list(range(num2_end, last_token+1))
                selected_X.append(X[key_indices, :])

            selected_X = np.stack(selected_X, axis=0)
            layer_scores = []
            for j in range(selected_X.shape[1]):
                Y_val = Y
                X_val = selected_X[:, j, :]

                if (args.logscale_prediction):
                    Y_train = np.log2(Y_train)
                else:
                    Y_train = Y_train
                regr = linears[target][i][j]
                score = regr.score(X_val, Y_val)
                layer_scores.append(score)
            
            scores[target].append(layer_scores)

    output_path = os.path.join(args.output_path, args.penalty)
    if not(os.path.exists(output_path)):
        os.makedirs(output_path)

    score_df = pd.DataFrame.from_dict(scores)
    with open(os.path.join(output_path, 'coefficient_scores_%s.txt' %(args.penalty)), 'w', encoding='utf-8') as fout:
        fout.write(str(score_df))

    # save linear probers
    for i in range(num_layers):
        for target in linears:
            full_path = os.path.join(output_path, 'layer_%d_prober_for_%s.bin' %(i, target))
            with open(full_path, 'wb') as fout:
                pickle.dump(linears[target][i], fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--control_path', type=str, default='./data/control.jsonl')
    parser.add_argument('--output_path', type=str, default='./model')
    parser.add_argument('--num_layers', type=int, default=32)
    parser.add_argument('--penalty', type=str, choices=['ridge', 'lasso', 'none'], default='ridge')
    parser.add_argument('--logscale_prediction', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.1)
    
    args = parser.parse_args()
    print(args)

    main(args)