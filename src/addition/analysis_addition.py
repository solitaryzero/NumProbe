import os
import argparse
import torch
import json
from tqdm import tqdm
import pickle
import time

import numpy as np
import pandas as pd
import sklearn.linear_model as linear_model
from sklearn.feature_selection import r_regression
from sklearn.metrics import r2_score

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_theme(style='ticks')
sns.set_context(rc={"axes.labelsize":20, "legend.fontsize":16, "legend.title_fontsize":16})
mpl.rcParams['lines.markersize'] = 6


def close_plots():
    plt.clf()
    plt.cla()
    plt.close('all')
    sns.set_context(rc={"axes.labelsize":20, "legend.fontsize":16, "legend.title_fontsize":16})
    time.sleep(0.1)


def savefig(fig, out_path, file_name, format):
    save_path = os.path.join(out_path, '%s.%s' %(file_name, format))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    fig.figure.savefig(save_path, format=format)
    close_plots()


def savegrid(grid, out_path, file_name, format):
    save_path = os.path.join(out_path, '%s.%s' %(file_name, format))
    grid.tick_params(labelsize=16)
    grid.fig.tight_layout(w_pad=1)
    grid.figure.savefig(save_path, format=format)
    close_plots()


def read_dataset(args):
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


def read_prober(args, penalty, layer, target):
    full_path = os.path.join(args.model_path, penalty, 'layer_%d_prober_for_%s.bin' %(layer, target))
    with open(full_path, 'rb') as fin:
        prober = pickle.load(fin)

    return prober


def analyze_addition(
    args,
    layer,
    test_dataset,
    test_num_examples,
):
    X = read_embeds(args, test_num_examples, args.split, layer)
    Y_a, Y_b, Y, G = test_dataset['a'], test_dataset['b'], test_dataset['predicted'], test_dataset['golden']
    model_a, model_b = read_prober(args, args.penalty, layer, 'a'), read_prober(args, args.penalty, layer, 'b')

    def predict(test_dataset, target, all_X, models):
        selected_X = []
        for j, X in enumerate(all_X):
            special_positions = test_dataset['special_positions'][j]
            tokens = test_dataset['tokens'][j]
            num1_start, num1_end, num2_start, num2_end, last_token = special_positions
            tokens[num1_end] = '<n1>'
            tokens[num2_end] = '<n2>'

            assert len(models) == (num2_start-num1_end)+(last_token-num2_end+1)

            if (target == 'a'):
                selected_X.append(X[num1_end, :])
                model = models[0]
            if (target == 'b'):
                selected_X.append(X[num2_end, :])
                model = models[num2_start-num1_end]

        X = np.stack(selected_X, axis=0)
        P = model.predict(X)
        return P
    
    P_a = predict(test_dataset, 'a', X, model_a)
    P_b = predict(test_dataset, 'b', X, model_b)

    P = np.round(np.exp2(P_a)+np.exp2(P_b))

    # LM_prediction = Y
    # AB_prediction = P
    valid_indexes = (Y != 0)
    Y, P, G = Y[valid_indexes], P[valid_indexes], G[valid_indexes]

    results = {
        'LM': {
            'accuracy': np.mean((Y == G)),
            'MSE': np.mean((Y-G)**2),
            'logMSE': np.mean((np.log2(Y)-np.log2(G))**2),
            'margin': np.min(np.array([np.max(np.abs((Y-G)/G)), 1])),
        },
        'AB': {
            'accuracy': np.mean((P == G)),
            'MSE': np.mean((P-G)**2),
            'logMSE': np.mean((np.log2(P)-np.log2(G))**2),
            'margin': np.min(np.array([np.max(np.abs((P-G)/G)), 1])),
        },
    }

    return results


def main(args):
    full_dataset, num_examples = read_dataset(args)
    test_dataset, test_num_examples = full_dataset[args.split], num_examples[args.split]

    if not(os.path.exists(args.output_path)):
        os.makedirs(args.output_path)
    for layer in tqdm(range(args.layer_count)):
        res = analyze_addition(args, layer, test_dataset, test_num_examples)
        with open(os.path.join(args.output_path, 'layer_%d.json' %layer), 'w', encoding='utf-8') as fout:
            json.dump(res, fout)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--output_path', type=str, default='./data/figures')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test')
    parser.add_argument('--penalty', type=str, choices=['ridge', 'lasso', 'none'], default='ridge')
    parser.add_argument('--layer_count', type=int, default=32)
    parser.add_argument('--logscale_prediction', action='store_true')

    args = parser.parse_args()
    print(args)

    main(args)