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


def read_prober(args, penalty, layer, target):
    full_path = os.path.join(args.model_path, penalty, 'layer_%d_prober_for_%s.bin' %(layer, target))
    with open(full_path, 'rb') as fin:
        prober = pickle.load(fin)

    return prober


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


def analyze_all_layers(
    args,
    test_dataset,
    num_test_examples,
):
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
            key_indices = list(range(num1_end, num2_start)) \
                        + list(range(num2_end, last_token+1))
            tokens[num1_end] = '<n1>'
            tokens[num2_end] = '<n2>'
            selected_X.append(X[key_indices, :])
            selected_tokens = [tokens[t] for t in key_indices]

        selected_X = np.stack(selected_X, axis=0)
        Y = test_dataset[args.target].to_numpy()

        models = read_prober(args, args.penalty, layer, target)
        for j in range(selected_X.shape[1]):
            # remove invalid prediction
            _X = selected_X[:, j, :]
            _Y = Y

            model = models[j]
            P = model.predict(_X)

            pearson = analyze_pearson(args, layer, target, _X, _Y, P)
            accuracy = analyze_accuracy(args, layer, _X, _Y, P)

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

    def beautify(fig):
        y_labels = fig.get_yticklabels()
        fig.set_yticklabels([selected_tokens[int(p.get_text())] for p in y_labels])
        fig.tick_params(axis='y', labelrotation=0)
        x_labels = fig.get_xticklabels()
        kept_labels = x_labels[::2]
        x_labels = ['' for x in x_labels]
        x_labels[::2] = kept_labels
        fig.set_xticklabels(x_labels)

    # Pearson correlation trends
    converted_df = df.pivot(index='position', columns='layer', values='pearson')
    fig = sns.heatmap(data=converted_df, annot=False, cmap='Blues')
    beautify(fig)
    savefig(fig, out_path, 'pearson_trend', args.format)

    # R2 Trends
    converted_df = df.pivot(index='position', columns='layer', values='r2')
    fig = sns.heatmap(data=converted_df, annot=False, cmap='Blues')
    beautify(fig)
    savefig(fig, out_path, 'r2_trend', args.format)

    # Accuracy trends
    converted_df = df.pivot(index='position', columns='layer', values='accuracy')
    fig = sns.heatmap(data=converted_df, annot=False, cmap='Blues')
    beautify(fig)
    savefig(fig, out_path, 'accuracy_trend', args.format)

    # MSE trends
    converted_df = df.pivot(index='position', columns='layer', values='mse')
    fig = sns.heatmap(data=converted_df, annot=False, cmap='Blues')
    beautify(fig)
    savefig(fig, out_path, 'mse_trend', args.format)

    # save gathered results
    with open(os.path.join(args.output_path, 'results.json'), 'w', encoding='utf-8') as fout:
        json.dump(gathered_results, fout)



def main(args):
    full_dataset, num_examples = read_dataset(args)
    test_dataset, test_num_examples = full_dataset[args.split], num_examples[args.split]

    op = args.output_path
    targets = ['control_signal']
    for t in targets:
        print('=== Analyzing target [%s] ===' %t)
        args.output_path = os.path.join(op, args.penalty, t)
        args.target = t

        analyze_all_layers(args, test_dataset, test_num_examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--control_path', type=str, default='./data/control.jsonl')
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