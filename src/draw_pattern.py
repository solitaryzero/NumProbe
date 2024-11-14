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
mpl.rcParams['lines.markersize'] = 2

from utils import read_dataset, read_embeds, read_prober


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


def draw_pattern(args, layer, target, position_token, X, Y, _P):
    if (args.logscale_prediction):
        Y = np.log2(Y)
        P = _P
    else:
        P = _P

    # # lm plot
    # df = pd.DataFrame({
    #     'log(Prediction)': P,
    #     'log(Golden)': Y,
    # })
    # fig = sns.lmplot(data=df, x='log(Golden)', y='log(Prediction)', line_kws={'color': '#e994a5'})

    # lm plot
    df = pd.DataFrame({
        'log(Prediction)': P,
        'log(Golden)': Y,
    })
    fig = sns.lmplot(data=df, x='log(Golden)', y='log(Prediction)', fit_reg=False)
    plt.plot([0, 35], [0, 35], color='#e994a5')

    out_path = os.path.join(args.output_path, position_token)
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    savefig(fig, out_path, 'layer_%d_target_%s' %(layer, target), args.format)


def draw_all_layers(
    args,
    test_dataset,
    num_test_examples,
):
    target = args.target
    for layer in tqdm(range(args.layer_count)):
        all_X = read_embeds(args, num_test_examples, 'test', layer)
        selected_X = []
        selected_tokens = []
        for j, X in enumerate(all_X):
            special_positions = test_dataset['special_positions'][j]
            num1_start, num1_end, num2_start, num2_end, last_token = special_positions
            key_indices = [num1_end, num2_end, last_token]
            probe_indices = list(range(num1_end, num2_start)) \
                            + list(range(num2_end, last_token+1))
            
            selected_probe_indices = [] # convert to probe indices
            for k, p in enumerate(probe_indices):
                if (p in key_indices):
                    selected_probe_indices.append(k)

            selected_X.append(X[key_indices, :])
            selected_tokens = ['n1', 'n2', 'last']

        selected_X = np.stack(selected_X, axis=0)
        Y = test_dataset[args.target].to_numpy()

        models = read_prober(args, args.penalty, layer, target)
        for j in range(selected_X.shape[1]):
            # remove invalid prediction
            if (target == 'predicted'):
                _X = selected_X[Y != 0, j, :]
                _Y = Y[Y != 0]
            else:
                _X = selected_X[:, j, :]
                _Y = Y

            model = models[selected_probe_indices[j]]
            P = model.predict(_X)

            draw_pattern(args, layer, target, selected_tokens[j], _X, _Y, P)


def main(args):
    full_dataset, num_examples = read_dataset(args)
    test_dataset, test_num_examples = full_dataset[args.split], num_examples[args.split]

    if (args.target != 'all'):
        args.output_path = os.path.join(args.output_path, 'patterns', args.target)
        draw_all_layers(args, test_dataset, test_num_examples)
    else:
        op = args.output_path
        targets = ['a', 'b', 'predicted']
        for t in targets:
            print('=== Analyzing target [%s] ===' %t)
            args.output_path = os.path.join(op, 'patterns', t)
            args.target = t

            draw_all_layers(args, test_dataset, test_num_examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--output_path', type=str, default='./data/figures')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test')
    parser.add_argument('--penalty', type=str, choices=['ridge', 'lasso', 'none'], default='ridge')
    parser.add_argument('--layer_count', type=int, default=32)
    parser.add_argument('--target', type=str, choices=['a', 'b', 'golden', 'predicted', 'all'], required=True)
    parser.add_argument('--logscale_prediction', action='store_true')

    args = parser.parse_args()
    print(args)
    
    main(args)