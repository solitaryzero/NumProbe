import os
import argparse
import json
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_theme(style='darkgrid')
sns.set_context(rc={"axes.labelsize":20, "legend.fontsize":16, "legend.title_fontsize":16})
mpl.rcParams['lines.markersize'] = 2

import numpy as np
import pandas as pd


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


def draw_main_results(args):
    models = ['llama-2-7b', 'llama-2-13b', 'Mistral-7B']
    layer_nums = {
        'llama-2-7b': 32,
        'Mistral-7B': 32,
        'llama-2-13b': 40,
    }
    targets = ['a', 'b', 'predicted']
    df_dict = {
        'model': [],
        'target': [],
    }

    def gather(args, target):
        for m in models:
            data_path = os.path.join(args.data_path, m, args.penalty, target, 'results.json')
            with open(data_path, 'r', encoding='utf-8') as fin:
                js = json.load(fin)
                n = len(js['layer'])
                for i in range(n):
                    if (target == 'a'):
                        if (js['token'][i] != '<n1>'):
                            continue
                    if (target == 'b'):
                        if (js['token'][i] != '<n2>'):
                            continue
                    if (target == 'predicted'):
                        if (i != n-1) and (js['token'][i+1] != '<n1>'):
                            continue

                    for key in js:
                        if (key == 'token'):
                            continue
                        if (key not in df_dict):
                            df_dict[key] = []

                        df_dict[key].append(js[key][i])
                    df_dict['model'].append(m)
                    df_dict['target'].append(target)

    if (args.target == 'all'):
        for t in targets:
            gather(args, t)
    else:
        gather(args, args.target)

    df_dict['layer_depth'] = []
    for i in range(len(df_dict['layer'])):
        df_dict['layer_depth'].append(df_dict['layer'][i]/layer_nums[df_dict['model'][i]])

    for key in df_dict:
        if (key != 'model') and (key != 'target'):
            df_dict[key] = np.array(df_dict[key])

    # data frames
    df = pd.DataFrame.from_dict(df_dict)

    out_path = os.path.join(args.out_path, 'merged')
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    for target in targets:
        df_target = df.loc[lambda df: df.target == target, :]
        # pearson figure
        fig = sns.lineplot(
            data=df_target, 
            x='layer_depth', 
            y='pearson', 
            hue='model', 
        )
        savefig(fig, out_path, 'pearson_trend_%s' %target, args.format)

        # r2 figure
        fig = sns.lineplot(
            data=df_target, 
            x='layer_depth', 
            y='r2', 
            hue='model', 
        )
        savefig(fig, out_path, 'r2_trend_%s' %target, args.format)

        # accuracy figure
        fig = sns.lineplot(
            data=df_target, 
            x='layer_depth', 
            y='accuracy', 
            hue='model', 
        )
        savefig(fig, out_path, 'accuracy_trend_%s' %target, args.format)

        # mse figure
        fig = sns.lineplot(
            data=df_target, 
            x='layer_depth', 
            y='mse', 
            hue='model', 
        )
        savefig(fig, out_path, 'mse_trend_%s' %target, args.format)


def draw_linearity(args):
    models = ['llama-2-7b_mlp', 'llama-2-7b']
    model_name_map = {
        'llama-2-7b': 'linear',
        'llama-2-7b_mlp': 'mlp',
    }
    layer_num = 32
    target = 'a'
    df_dict = {
        'probe': [],
        'target': [],
    }

    def gather(args, target):
        for m in models:
            if ('mlp' in m):
                data_path = os.path.join(args.data_path, m, target, 'results.json')
                with open(data_path, 'r', encoding='utf-8') as fin:
                    js = json.load(fin)
                    layer_num = None
                    for key in js:
                        if (layer_num is None):
                            layer_num = len(js[key])
                        else:
                            assert (len(js[key]) == layer_num)

                        if (key not in df_dict):
                            df_dict[key] = []

                        df_dict[key].extend(js[key])
                    df_dict['probe'].extend([model_name_map[m] for _ in range(layer_num)])
                    df_dict['target'].extend([target for _ in range(layer_num)])
            else:
                data_path = os.path.join(args.data_path, m, args.penalty, target, 'results.json')
                with open(data_path, 'r', encoding='utf-8') as fin:
                    js = json.load(fin)
                    n = len(js['layer'])
                    for i in range(n):
                        if (target == 'a'):
                            if (js['token'][i] != '<n1>'):
                                continue
                        if (target == 'b'):
                            if (js['token'][i] != '<n2>'):
                                continue
                        if (target == 'predicted'):
                            if (i != n-1) and (js['token'][i+1] != '<n1>'):
                                continue

                        for key in js:
                            if (key not in df_dict):
                                continue

                            df_dict[key].append(js[key][i])
                        df_dict['probe'].append(model_name_map[m])
                        df_dict['target'].append(target)

    gather(args, target)

    df_dict['layer_depth'] = []
    for i in range(len(df_dict['layer'])):
        df_dict['layer_depth'].append(df_dict['layer'][i]/layer_num)

    for key in df_dict:
        if (key != 'probe') and (key != 'target'):
            df_dict[key] = np.array(df_dict[key])

    # data frames
    df = pd.DataFrame.from_dict(df_dict)

    out_path = os.path.join(args.data_path, 'linearity')
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    # pearson figure
    fig = sns.lineplot(
        # data=df, 
        data=df.loc[lambda df: df.pearson > 0.8, :], 
        x='layer_depth', 
        y='pearson', 
        hue='probe', 
    )
    savefig(fig, out_path, 'linear_pearson_trend', args.format)

    # r2 figure
    fig = sns.lineplot(
        # data=df, 
        data=df.loc[lambda df: df.r2 > 0.8, :], 
        x='layer_depth', 
        y='r2', 
        hue='probe',
    )
    savefig(fig, out_path, 'linear_r2_trend', args.format)

    # accuracy figure
    fig = sns.lineplot(
        data=df, 
        x='layer_depth', 
        y='accuracy', 
        hue='probe', 
    )
    savefig(fig, out_path, 'linear_accuracy_trend', args.format)

    # mse figure
    fig = sns.lineplot(
        data=df, 
        x='layer_depth', 
        y='mse', 
        hue='probe', 
    )
    savefig(fig, out_path, 'linear_mse_trend', args.format)


def draw_coefficient(args):
    models = ['llama-2-7b', 'llama-2-13b', 'Mistral-7B']
    layer_num = {
        'llama-2-7b': 32,
        'Mistral-7B': 32,
        'llama-2-13b': 40,
    }
    targets = ['a']
    df_dict = {
        'model': [],
        'target': [],
    }

    def gather(args, target):
        for m in models:
            data_path = os.path.join(args.data_path, m, args.penalty, target, 'coefficients.json')
            with open(data_path, 'r', encoding='utf-8') as fin:
                js = json.load(fin)
                layer_num = None
                for key in js:
                    if (layer_num is None):
                        layer_num = len(js[key])
                    else:
                        assert (len(js[key]) == layer_num)

                    if (key not in df_dict):
                        df_dict[key] = []

                    df_dict[key].extend(js[key])
                df_dict['model'].extend([m for _ in range(layer_num)])
                df_dict['target'].extend([target for _ in range(layer_num)])

    if (args.target == 'all'):
        for t in targets:
            gather(args, t)
    else:
        gather(args, args.target)

    df_dict['layer_depth'] = []
    for i in range(len(df_dict['layer'])):
        df_dict['layer_depth'].append(df_dict['layer'][i]/layer_num[df_dict['model'][i]])

    for key in df_dict:
        if (key != 'model') and (key != 'target') and (key != 'feature'):
            df_dict[key] = np.array(df_dict[key])

    # data frames
    df = pd.DataFrame.from_dict(df_dict)
    df_wo_0 = df.loc[lambda df: df.layer != 0, :]

    out_path = os.path.join(args.out_path, 'merged')
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    # pearson figure
        # Initialize a grid of plots with an Axes for each walk
    grid = sns.FacetGrid(df, col="feature", hue="model", 
                     col_wrap=4, height=4)

    # Draw a line plot to show the trajectory
    grid.map(plt.plot, "layer", "value", marker="o")

    # Adjust the arrangement of the plots
    grid.fig.tight_layout(w_pad=1)
    grid.add_legend()

    savegrid(grid, out_path, 'coefficient', args.format)


def draw_accuracy(args):
    models = ['llama-2-7b', 'llama-2-13b', 'Mistral-7B']
    digit_count = 9
    target = 'a'
    df_dict = {
        'model': [],
    }

    def gather(args, target):
        for m in models:
            data_path = os.path.join(args.data_path, m, args.penalty, target, 'overall_accuracy.json')
            with open(data_path, 'r', encoding='utf-8') as fin:
                js = json.load(fin)
                for key in js:
                    if (key not in df_dict):
                        df_dict[key] = []

                    df_dict[key].extend(js[key])
                df_dict['model'].extend([m for _ in range(digit_count)])

    gather(args, target)

    for key in df_dict:
        if (key != 'model'):
            df_dict[key] = np.array(df_dict[key])

    # data frames
    df = pd.DataFrame.from_dict(df_dict)

    out_path = os.path.join(args.out_path, 'overall_accuracy')
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    fig = sns.lineplot(
        data=df, 
        x='digits', 
        y='accuracy', 
        hue='model', 
    )
    savefig(fig, out_path, 'overall_accuracy', args.format)


def draw_addition(args):
    models = ['llama-2-7b_8', 'llama-2-13b_8', 'Mistral-7B_8']
    layer_num = {
        'llama-2-7b_8': 32,
        'Mistral-7B_8': 32,
        'llama-2-13b_8': 40,
    }

    df_dict = {
        'model': [],
        'layer': [],
        'type': [],
    }

    def gather(args, model, layer):
        data_path = os.path.join(args.data_path, model, 'addition', 'layer_%d.json' %layer)
        with open(data_path, 'r', encoding='utf-8') as fin:
            js = json.load(fin)
            for type in js:
                for key in js[type]:
                    if (key not in df_dict):
                        df_dict[key] = []

                    df_dict[key].append(js[type][key])

                df_dict['model'].append(model)
                df_dict['layer'].append(layer)
                df_dict['type'].append(type)

    for model in models:
        for layer in range(1, layer_num[model]):
            gather(args, model, layer)

    df_dict['layer_depth'] = []
    for i in range(len(df_dict['layer'])):
        df_dict['layer_depth'].append(df_dict['layer'][i]/layer_num[df_dict['model'][i]])

    for key in df_dict:
        if (key not in ['model', 'type']):
            df_dict[key] = np.array(df_dict[key])

    # data frames
    df = pd.DataFrame.from_dict(df_dict)

    out_path = os.path.join(args.out_path, 'addition')
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    # log(MSE) figure
    fig = sns.lineplot(
        data=df, 
        x='layer_depth', 
        y='logMSE', 
        hue='model', 
        style='type',
    )
    savefig(fig, out_path, 'logMSE_8', args.format)

    # error margin figure
    fig = sns.lineplot(
        data=df, 
        x='layer_depth', 
        y='margin', 
        hue='model', 
        style='type',
    )
    savefig(fig, out_path, 'margin_8', args.format)


def draw_fine(args):
    draw_addition(args)


def main(args):
    tasks = args.tasks.split(',')
    if ('main' in tasks):
        draw_main_results(args)
    if ('linearity' in tasks):
        draw_linearity(args)
    if ('coeff' in tasks):
        draw_coefficient(args)
    if ('accuracy' in tasks):
        draw_accuracy(args)
    if ('fine' in tasks):
        draw_fine(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/figures')
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--out_path', type=str, default='./data/figures')
    parser.add_argument('--penalty', type=str, choices=['ridge', 'lasso', 'none'], default='ridge')
    parser.add_argument('--target', type=str, choices=['a', 'b', 'golden', 'predicted_answer', 'all'], required=True)
    parser.add_argument('--tasks', type=str, default='main,coeff,accuracy,fine')

    args = parser.parse_args()
    print(args)

    main(args)