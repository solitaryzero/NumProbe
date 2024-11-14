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


def draw_partial(args):
    models = ['llama-2-7b_8d', 'llama-2-13b_8d', 'Mistral-7B_8d']
    layer_nums = {
        'llama-2-7b_8d': 32,
        'Mistral-7B_8d': 32,
        'llama-2-13b_8d': 40,
    }
    target = 'a'
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
                    for key in js:
                        if (key == 'token'):
                            continue
                        if (key not in df_dict):
                            df_dict[key] = []

                        df_dict[key].append(js[key][i])
                    df_dict['model'].append(m)
                    df_dict['target'].append(target)

    gather(args, target)

    df_dict['layer_depth'] = []
    for i in range(len(df_dict['layer'])):
        df_dict['layer_depth'].append(df_dict['layer'][i]/layer_nums[df_dict['model'][i]])

    for key in df_dict:
        if (key != 'model') and (key != 'target'):
            df_dict[key] = np.array(df_dict[key])

    # data frames
    raw_df = pd.DataFrame.from_dict(df_dict)

    for model in models:
        out_path = os.path.join(args.out_path, model)
        if not(os.path.exists(out_path)):
            os.makedirs(out_path)

        df = raw_df.loc[lambda df: df.model == model, :]

        def beautify(fig):
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


def main(args):
    draw_partial(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/figures')
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--out_path', type=str, default='./data/figures')
    parser.add_argument('--penalty', type=str, choices=['ridge', 'lasso', 'none'], default='ridge')
    args = parser.parse_args()
    print(args)

    main(args)