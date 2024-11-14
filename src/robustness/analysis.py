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
sns.set_theme(style='darkgrid')
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

    return np.stack(embeddings, axis=0)


def read_prober(args, penalty, layer, target):
    full_path = os.path.join(args.model_path, penalty, 'layer_%d_prober_for_%s.bin' %(layer, target))
    with open(full_path, 'rb') as fin:
        prober = pickle.load(fin)

    prober = prober[0]
    return prober


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
    )
    print('Pearson Correlation of layer %d: %.4f' %(layer, pearson_correlation))

    # lm plot
    df = pd.DataFrame({
        'log(Prediction)': P,
        'log(Golden)': Y,
    })
    fig = sns.lmplot(data=df, x='log(Golden)', y='log(Prediction)', line_kws={'color': '#e994a5'})

    out_path = os.path.join(args.output_path, 'pearson')
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    savefig(fig, out_path, 'pearson_layer_%d_target_%s' %(layer, target), args.format)

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

    correct = np.sum(Y == P)
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
        digit, golden, prediction = row['digit'], row['golden'], row['predicted_answer']
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

    gathered_coefficients = {
        'layer': [],
        'value': [],
        'feature': [],
    }

    # filter template type
    slice = test_dataset[test_dataset['template_type'] == args.template_type]
    index_list = slice.index.tolist()

    target = args.target
    for layer in range(args.layer_count):
        X = read_embeds(args, test_num_examples, args.split, layer)
        Y = test_dataset[args.target]

        # apply filter
        X = X[index_list]
        Y = Y[index_list]

        Y = Y.to_numpy()

        model = read_prober(args, args.penalty, layer, target)
        P = model.predict(X)

        pearson = analyze_pearson(args, layer, target, X, Y, P)
        accuracy = analyze_accuracy(args, layer, X, Y, P)
        coefficients = analyze_coefficients(args, layer, model)

        gathered_results['layer'].append(pearson['layer'])
        gathered_results['r2'].append(pearson['r2'])
        gathered_results['pearson'].append(pearson['pearson'])
        gathered_results['accuracy'].append(accuracy['accuracy'])
        gathered_results['mse'].append(accuracy['mse'])

        for key in coefficients:
            if (key in ['max_value', 'min_value', 'average', 'variance']):
                gathered_coefficients['layer'].append(layer)
                gathered_coefficients['feature'].append(key)
                gathered_coefficients['value'].append(coefficients[key])


    df = pd.DataFrame.from_dict(gathered_results)
    df_wo0 = df[df['layer'] != 0]
    out_path = args.output_path
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    # Pearson correlation trends
    fig = sns.lineplot(data=df_wo0, x='layer', y='pearson')
    savefig(fig, out_path, 'pearson_trend', args.format)

    # R2 Trends
    fig = sns.lineplot(data=df_wo0, x='layer', y='r2')
    savefig(fig, out_path, 'r2_trend', args.format)

    # Accuracy trends
    fig = sns.lineplot(data=df, x='layer', y='accuracy')
    savefig(fig, out_path, 'accuracy_trend', args.format)

    fig = sns.lineplot(data=df, x='layer', y='mse')
    savefig(fig, out_path, 'mse_trend', args.format)

    # coeffcient trends
    df = pd.DataFrame.from_dict(gathered_coefficients)

    # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col="feature", hue="feature", palette="tab20c",
                     col_wrap=2, height=4)
    grid.map(plt.plot, "layer", "value", marker="o")

    # Adjust the arrangement of the plots
    grid.fig.tight_layout(w_pad=1)
    savegrid(grid, out_path, 'coefficient_trend', args.format)


    # save gathered results
    with open(os.path.join(args.output_path, 'results.json'), 'w', encoding='utf-8') as fout:
        json.dump(gathered_results, fout)

    with open(os.path.join(args.output_path, 'coefficients.json'), 'w', encoding='utf-8') as fout:
        gc = {}
        for k in gathered_coefficients:
            gc[k] = gathered_coefficients[k]
        json.dump(gathered_coefficients, fout)



def main(args):
    template_num = 2

    full_dataset, num_examples = read_dataset(args)
    test_dataset, test_num_examples = full_dataset[args.split], num_examples[args.split]

    out_base_path = args.output_path
    for tem_t in range(template_num):
        args.output_path = os.path.join(out_base_path, 'prompt_%d' %tem_t, args.target)
        args.template_type = tem_t

        analyze_all_layers(args, test_dataset, test_num_examples)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--output_path', type=str, default='./data/figures')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test')
    parser.add_argument('--penalty', type=str, choices=['ridge', 'lasso', 'none'], default='ridge')
    parser.add_argument('--layer_count', type=int, default=32)
    parser.add_argument('--template_type', type=int)
    parser.add_argument('--target', type=str, choices=['a'], required=True)
    parser.add_argument('--logscale_prediction', action='store_true')

    args = parser.parse_args()
    print(args)
    
    main(args)