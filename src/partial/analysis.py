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
    plt.xticks(fontsize=14)
    plt.yticks(fontsize=10)
    plt.tight_layout()
    fig.figure.savefig(save_path, format=format)
    close_plots()


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


def analyze_coefficients(args, layer, model):
    coefficients = model.coef_
    return {
        'layer': layer,
        'max_value': np.max(coefficients).item(),
        'max_pos': np.argmax(coefficients).item(),
        'min_value': np.min(coefficients).item(),
        'min_pos': np.argmin(coefficients).item(),
        'average': np.average(coefficients).item(),
        'variance': np.var(coefficients).item(),
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
    savefig(fig, out_path, 'overall_accuracy', args.format)

    # save overall accuracy
    with open(os.path.join(args.output_path, 'overall_accuracy.json'), 'w', encoding='utf-8') as fout:
        json.dump({'digits': digits, 'accuracy': accuracy}, fout)


def analyze_one_layer(
    args,
    layer,
    target,
    test_dataset,
    num_test_examples,
):
    X = read_embeds(args, num_test_examples, args.split, layer)
    Y = test_dataset[args.target]
    # remove invalid prediction
    if (target == 'predicted_answer'):
        X = X[Y != 0]
        Y = Y[Y != 0]
    Y = Y.to_numpy()

    model = read_prober(args, args.penalty, layer, target)
    P = model.predict(X)

    print(analyze_pearson(args, layer, target, X, Y, P))
    print(analyze_accuracy(args, layer, X, Y, P))


def analyze_all_layers(
    args,
    test_dataset,
    num_test_examples,
):
    analyze_overall_accuracy(args, test_dataset)

    gathered_results = {
        'layer': [],
        'position': [],
        'token': [],
        'pearson': [],
        'r2': [],
        'accuracy': [],
        'mse': [],
    }

    target = args.target
    for layer in tqdm(range(args.layer_count)):
        all_X = read_embeds(args, num_test_examples, 'test', layer)
        selected_X = []
        selected_tokens = []
        for j, X in enumerate(all_X):
            special_positions = test_dataset['special_positions'][j]
            tokens = test_dataset['tokens'][j]
            num1_start, num1_end, num2_start, num2_end, last_token = special_positions
            key_indices = list(range(num1_start, num1_end+1))
            selected_X.append(X[key_indices, :])
            selected_tokens = [tokens[t] for t in key_indices]

        selected_X = np.stack(selected_X, axis=0)
        Y = test_dataset[args.target].to_numpy()

        models = read_prober(args, args.penalty, layer, target)
        for j in range(selected_X.shape[1]):
            # remove invalid prediction
            _X = selected_X[:, j, :]
            _Y = np.array([int(str(y)[:j+1]) for y in Y])

            model = models[j]
            P = model.predict(_X)

            pearson = analyze_pearson(args, layer, target, _X, _Y, P)
            accuracy = analyze_accuracy(args, layer, _X, _Y, P)
            # coefficients = analyze_coefficients(args, layer, model)

            gathered_results['layer'].append(layer)
            gathered_results['position'].append(j)
            gathered_results['token'].append(selected_tokens[j])
            gathered_results['r2'].append(pearson['r2'])
            gathered_results['pearson'].append(pearson['pearson'])
            gathered_results['accuracy'].append(accuracy['accuracy'])
            gathered_results['mse'].append(accuracy['mse'])


    df = pd.DataFrame.from_dict(gathered_results)
    out_path = args.output_path
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    # Pearson correlation trends
    converted_df = df.pivot(index='layer', columns='position', values='pearson')
    fig = sns.heatmap(data=converted_df, annot=False)
    savefig(fig, out_path, 'pearson_trend', args.format)

    # R2 Trends
    converted_df = df.pivot(index='layer', columns='position', values='r2')
    fig = sns.heatmap(data=converted_df, annot=False)
    savefig(fig, out_path, 'r2_trend', args.format)

    # Accuracy trends
    converted_df = df.pivot(index='layer', columns='position', values='accuracy')
    fig = sns.heatmap(data=converted_df, annot=False)
    savefig(fig, out_path, 'accuracy_trend', args.format)

    # MSE trends
    converted_df = df.pivot(index='layer', columns='position', values='mse')
    fig = sns.heatmap(data=converted_df, annot=False)
    savefig(fig, out_path, 'mse_trend', args.format)

    # save gathered results
    with open(os.path.join(args.output_path, 'results.json'), 'w', encoding='utf-8') as fout:
        json.dump(gathered_results, fout)



def main(args):
    full_dataset, num_examples = read_dataset(args)
    test_dataset, test_num_examples = full_dataset[args.split], num_examples[args.split]

    args.output_path = os.path.join(args.output_path, args.penalty, args.target)
    if (args.layer == -1):
        analyze_all_layers(args, test_dataset, test_num_examples)
    else:
        analyze_one_layer(args, args.layer, args.target, test_dataset, test_num_examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--output_path', type=str, default='./data/figures')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test')
    parser.add_argument('--penalty', type=str, choices=['ridge', 'lasso', 'none'], default='ridge')
    parser.add_argument('--layer', type=int, required=True)
    parser.add_argument('--layer_count', type=int, default=32)
    parser.add_argument('--target', type=str, choices=['a'], required=True)
    parser.add_argument('--logscale_prediction', action='store_true')

    args = parser.parse_args()
    print(args)
    
    main(args)