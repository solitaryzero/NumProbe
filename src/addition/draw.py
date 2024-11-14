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


def draw_addition(args):
    models = ['llama-2-7b', 'llama-2-13b', 'Mistral-7B']
    layer_num = {
        'llama-2-7b': 32,
        'llama-2-13b': 40,
        'Mistral-7B': 32,
    }

    df_dict = {
        'model': [],
        'layer': [],
        'type': [],
    }

    def gather(args, model, layer):
        data_path = os.path.join(args.data_path, model, 'layer_%d.json' %layer)
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
        for layer in range(layer_num[model]):
            gather(args, model, layer)

    df_dict['layer_depth'] = []
    for i in range(len(df_dict['layer'])):
        df_dict['layer_depth'].append(df_dict['layer'][i]/layer_num[df_dict['model'][i]])

    for key in df_dict:
        if (key not in ['model', 'type']):
            df_dict[key] = np.array(df_dict[key])

    # data frames
    df = pd.DataFrame.from_dict(df_dict)

    out_path = os.path.join(args.data_path, 'addition')
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

    savefig(fig, out_path, 'logMSE', args.format)

    # error margin figure
    fig = sns.lineplot(
        data=df, 
        x='layer_depth', 
        y='margin', 
        hue='model', 
        style='type',
    )

    savefig(fig, out_path, 'margin', args.format)


def main(args):
    draw_addition(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/figures/addition')
    parser.add_argument('--out_path', type=str, default='./data/figures/addition')
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--penalty', type=str, choices=['ridge', 'lasso', 'none'], default='ridge')

    args = parser.parse_args()
    print(args)

    main(args)