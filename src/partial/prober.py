import os
import argparse
import torch
import json
from tqdm import tqdm
import pickle

import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model

from utils import read_dataset, read_embeds

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

    target = 'a'
    linears = []

    for i in tqdm(range(num_layers)): # ignore raw embedding layer
        Y = train_dataset[target]

        all_X = read_embeds(args, num_train_examples, 'train', i)
        selected_X = []
        for j, X in enumerate(all_X):
            special_positions = train_dataset['special_positions'][j]
            num1_start, num1_end, num2_start, num2_end, last_token = special_positions
            key_indices = list(range(num1_start, num1_end+1))
            selected_X.append(X[key_indices, :])

        selected_X = np.stack(selected_X, axis=0)
        layer_regressors = []
        for j in range(selected_X.shape[1]):
            Y_train = np.array([int(str(y)[:j+1]) for y in Y])
            X_train = selected_X[:, j, :]
            
            if (args.logscale_prediction):
                Y_train = np.log2(Y_train)
            else:
                Y_train = Y_train
            regr = model_factory(args)

            # print(X_train.shape)
            # print(Y_train.shape)
            regr.fit(X_train, Y_train)
            # print(regr.coef_)
            # input()

            layer_regressors.append(regr)
        
        linears.append(layer_regressors)

    # score on validation set
    valid_dataset = full_dataset['val']
    num_valid_examples = num_examples['val']

    # calculate score on the valid set
    scores = []

    for i in tqdm(range(num_layers)): # ignore raw embedding layer
        Y = valid_dataset[target]

        all_X = read_embeds(args, num_valid_examples, 'val', i)
        selected_X = []
        for j, X in enumerate(all_X):
            special_positions = valid_dataset['special_positions'][j]
            num1_start, num1_end, num2_start, num2_end, last_token = special_positions
            key_indices = list(range(num1_start, num1_end+1))
            selected_X.append(X[key_indices, :])

        selected_X = np.stack(selected_X, axis=0)
        layer_scores = []
        for j in range(selected_X.shape[1]):
            Y_val = np.array([int(str(y)[:j+1]) for y in Y])
            X_val = selected_X[:, j, :]

            if (args.logscale_prediction):
                Y_val = np.log2(Y_val)
            else:
                Y_val = Y_val

            regr = linears[i][j]
            score = regr.score(X_val, Y_val)
            layer_scores.append(score)
        
        scores.append(layer_scores)

    output_path = os.path.join(args.output_path, args.penalty)
    if not(os.path.exists(output_path)):
        os.makedirs(output_path)

    with open(os.path.join(output_path, 'coefficient_scores.json'), 'w', encoding='utf-8') as fout:
        fout.write(json.dumps(scores, indent=4))

    # save linear probers
    for i in range(num_layers):
        for target in linears:
            full_path = os.path.join(output_path, 'layer_%d_prober.bin' %(i))
            with open(full_path, 'wb') as fout:
                pickle.dump(linears[i], fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--output_path', type=str, default='./model')
    parser.add_argument('--num_layers', type=int, default=33)
    parser.add_argument('--penalty', type=str, choices=['ridge', 'lasso', 'none'], default='ridge')
    parser.add_argument('--logscale_prediction', action='store_true')
    parser.add_argument('--alpha', type=float, default=0.1)
    
    args = parser.parse_args()
    print(args)

    main(args)