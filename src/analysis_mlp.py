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
mpl.rcParams['lines.markersize'] = 2

from prober_mlp import MLPProber
from utils import read_dataset, read_embeds, read_prober


def close_plots():
    plt.clf()
    plt.cla()
    plt.close('all')
    time.sleep(0.1)


def analyze_pearson(args, layer, target, X, Y, _P):
    if (args.logscale_prediction):
        Y = np.log2(Y)
        P = _P
    else:
        P = _P

    # Pearson Correlation
    pearson_correlation = r_regression(
        X=P.reshape(-1, 1),
        y=Y,
        center=False,
        force_finite=True,
    )
    # print('Pearson Correlation of layer %d: %.4f' %(layer, pearson_correlation))

    r2 = r2_score(Y, P)

    return {
        'layer': layer,
        'pearson': pearson_correlation.item(),
        'r2': r2.item(),
    }


def analyze_accuracy(args, layer, X, Y, _P):
    if (args.logscale_prediction):
        P = np.exp2(_P)
        P = (np.rint(P)).astype(int)
    else:
        P = (np.rint(_P)).astype(int)

    # correct = np.sum(Y == P)
    correct = np.sum(np.abs(P-Y) <= (Y*0.01))
    total = P.shape[0]
    accuracy = correct / total

    if (args.logscale_prediction):
        _Y = np.log2(Y)
        mse = np.mean((_Y-_P)**2)
    else:
        mse = np.mean((Y-P)**2)

    return {
        'layer': layer,
        'accuracy': accuracy,
        'mse': mse,
    }


def analyze_overall_accuracy(args, test_dataset):
    total, correct = {}, {}

    for index, row in test_dataset.iterrows():
        digit, golden, prediction = row['digit'], row['golden'], row['predicted']
        total[digit] = total.get(digit, 0)+1
        if (golden == prediction):
            correct[digit] = correct.get(digit, 0)+1

    digits, accuracy = [], []
    for digit in total:
        digits.append(digit)
        accuracy.append(correct.get(digit, 0)/total[digit])

    df = pd.DataFrame({
        'Digits': np.array(digits),
        'Accuracy': np.array(accuracy),
    })
    fig = sns.lineplot(data=df, x='Digits', y='Accuracy')

    out_path = os.path.join(args.output_path, 'overall_accuracy')
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    save_path = os.path.join(out_path, 'overall_accuracy.png')
    fig.figure.savefig(save_path)
    close_plots()


def analyze_all_layers(
    args,
    test_dataset,
    test_num_examples,
):
    analyze_overall_accuracy(args, test_dataset)

    gathered_results = {
        'layer': [],
        'pearson': [],
        'r2': [],
        'accuracy': [],
        'mse': [],
    }

    pos_map = {
        'a': 1, # num1 end
        'b': 3, # num2 end
        # 'predicted': 4, # last token
    }

    target = args.target
    for layer in range(args.layer_count):
        all_X = read_embeds(args, test_num_examples, args.split, layer)
        Y = test_dataset[args.target]
        Y = Y.to_numpy()

        X = []
        for j, _X in enumerate(all_X):
            special_positions = test_dataset['special_positions'][j]
            X.append(_X[special_positions[pos_map[target]], :]) 
        X = np.stack(X)

        params = args.__dict__
        params['load_model_path'] = os.path.join(args.model_path, 'layer_%d_prober_%s' %(layer, target), 'checkpoint.bin')
        model = MLPProber(params)
        with torch.no_grad():
            P = model.predict(torch.FloatTensor(X)).numpy()

        pearson = analyze_pearson(args, layer, target, X, Y, P)
        accuracy = analyze_accuracy(args, layer, X, Y, P)

        gathered_results['layer'].append(pearson['layer'])
        gathered_results['pearson'].append(pearson['pearson'])
        gathered_results['r2'].append(pearson['r2'])
        gathered_results['accuracy'].append(accuracy['accuracy'])
        gathered_results['mse'].append(accuracy['mse'])


    df = pd.DataFrame.from_dict(gathered_results)
    # Pearson correlation trends
    fig = sns.lineplot(data=df, x='layer', y='pearson')

    out_path = args.output_path
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    save_path = os.path.join(out_path, 'pearson_trend.png')
    fig.figure.savefig(save_path)
    close_plots()

    # R2 Trends
    fig = sns.lineplot(data=df, x='layer', y='r2')

    out_path = args.output_path
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    save_path = os.path.join(out_path, 'r2_trend.png')
    fig.figure.savefig(save_path)
    close_plots()

    # Accuracy trends
    fig = sns.lineplot(data=df, x='layer', y='accuracy')

    out_path = args.output_path
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    save_path = os.path.join(out_path, 'accuracy_trend.png')
    fig.figure.savefig(save_path)
    close_plots()

    fig = sns.lineplot(data=df, x='layer', y='mse')

    out_path = args.output_path
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    save_path = os.path.join(out_path, 'mse_trend.png')
    fig.figure.savefig(save_path)
    close_plots()

    # save gathered results
    with open(os.path.join(args.output_path, 'results.json'), 'w', encoding='utf-8') as fout:
        json.dump(gathered_results, fout)


def main(args):
    full_dataset, num_examples = read_dataset(args)
    test_dataset, test_num_examples = full_dataset[args.split], num_examples[args.split]

    analyze_all_layers(args, test_dataset, test_num_examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--output_path', type=str, default='./data/figures')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test')
    parser.add_argument('--layer_count', type=int, default=32)
    parser.add_argument('--hidden_size', type=int, default=4096)
    parser.add_argument('--prober_dim', type=int, default=256)
    parser.add_argument('--target', type=str, choices=['a', 'b', 'golden', 'predicted', 'all'], required=True)
    parser.add_argument('--logscale_prediction', action='store_true')

    args = parser.parse_args()
    args.output_path = os.path.join(args.output_path, args.target)
    print(args)

    main(args)