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


def draw_difference(args):
    models = ['llama-2-7b', 'llama-2-13b', 'Mistral-7B']
    layer_nums = {
        'llama-2-7b': 32,
        'Mistral-7B': 32,
        'llama-2-13b': 40,
    }
    targets = ['a', 'b']
    metrics = ['pearson', 'r2', 'accuracy', 'mse']
    df_dict = {
        'model': [],
        'target': [],
    }
    selected_tokens = {}

    # get results on control signal
    control_results = {}
    for m in models:
        data_path = os.path.join(args.data_path, 'control', m, args.penalty, 'control_signal', 'results.json')
        with open(data_path, 'r', encoding='utf-8') as fin:
            js = json.load(fin)
            n = len(js['layer'])
            for i in range(n):
                dict_key = (m, js['layer'][i], js['position'][i])
                control_results[dict_key] = {}
                for key in js:
                    if (key in metrics):
                        control_results[dict_key][key] = js[key][i]


    def gather(args, target):
        for m in models:
            data_path = os.path.join(args.data_path, m, args.penalty, target, 'results.json')
            selected_tokens[m] = {}
            with open(data_path, 'r', encoding='utf-8') as fin:
                js = json.load(fin)
                n = len(js['layer'])
                for i in range(n):
                    for key in js:
                        if (key not in df_dict):
                            df_dict[key] = []

                        if (key in metrics):
                            dict_key = (m, js['layer'][i], js['position'][i])
                            control_value = control_results[dict_key][key]
                            df_dict[key].append(js[key][i]-control_value)
                        else:
                            df_dict[key].append(js[key][i])
                    df_dict['model'].append(m)
                    df_dict['target'].append(target)
                    selected_tokens[m][js['position'][i]] = js['token'][i]

    if (args.target == 'all'):
        for t in targets:
            gather(args, t)
    else:
        gather(args, args.target)

    for key in df_dict:
        if (key != 'model') and (key != 'target') and (key != 'token'):
            df_dict[key] = np.array(df_dict[key])

    # data frames
    df = pd.DataFrame.from_dict(df_dict)

    for model in models:
        for target in targets:
            out_path = os.path.join(args.out_path, 'control', 'merged', model)
            if not(os.path.exists(out_path)):
                os.makedirs(out_path)

            df_target = df.loc[lambda df: (df.model == model), :]
            df_target = df_target.loc[lambda df: (df.target == target), :]
            def beautify(fig):
                y_labels = fig.get_yticklabels()
                fig.set_yticklabels([selected_tokens[model][int(p.get_text())] for p in y_labels])
                fig.tick_params(axis='y', labelrotation=0)
                x_labels = fig.get_xticklabels()
                kept_labels = x_labels[::2]
                x_labels = ['' for x in x_labels]
                x_labels[::2] = kept_labels
                fig.set_xticklabels(x_labels)

            # Pearson correlation trends
            converted_df = df_target.pivot(index='position', columns='layer', values='pearson')
            fig = sns.heatmap(data=converted_df, annot=False, cmap='Blues')
            beautify(fig)
            savefig(fig, out_path, 'pearson_trend_%s' %target, args.format)

            # R2 Trends
            converted_df = df_target.pivot(index='position', columns='layer', values='r2')
            fig = sns.heatmap(data=converted_df, annot=False, cmap='Blues')
            beautify(fig)
            savefig(fig, out_path, 'r2_trend_%s' %target, args.format)

            # Accuracy trends
            converted_df = df_target.pivot(index='position', columns='layer', values='accuracy')
            fig = sns.heatmap(data=converted_df, annot=False, cmap='Blues')
            beautify(fig)
            savefig(fig, out_path, 'accuracy_trend_%s' %target, args.format)

            # MSE trends
            converted_df = df_target.pivot(index='position', columns='layer', values='mse')
            fig = sns.heatmap(data=converted_df, annot=False, cmap='Blues')
            beautify(fig)
            savefig(fig, out_path, 'mse_trend_%s' %target, args.format)


def main(args):
    tasks = args.tasks.split(',')
    if ('main' in tasks):
        draw_difference(args)


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