import os
import argparse
import json
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_theme(style='darkgrid', palette='Set2')
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


def draw_robust(args):
    models = ['llama-2-7b']
    # template_num = 2
    template_num = 1
    layer_num = {
        'llama-2-7b': 32,
        'llama-2-13b': 40,
    }
    df_dict = {
        'model': [],
        'template_type': [],
        'target': [],
    }

    def gather(args, model, target, t_type, data_path):
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
            df_dict['model'].extend([model for _ in range(layer_num)])
            df_dict['template_type'].extend([t_type for _ in range(layer_num)])
            df_dict['target'].extend([target for _ in range(layer_num)])


    def gather_raw(args, model, target, t_type, data_path):
        with open(data_path, 'r', encoding='utf-8') as fin:
            js = json.load(fin)
            n = len(js['layer'])
            for i in range(n):
                if (js['token'][i] != '<n1>'):
                    continue

                for key in js:
                    if (key == 'token') or (key == 'position'):
                        continue
                    if (key not in df_dict):
                        df_dict[key] = []

                    df_dict[key].append(js[key][i])
                df_dict['model'].append(model)
                df_dict['template_type'].append(t_type)
                df_dict['target'].append(target)

    t = args.target
    for m in models:
        # original
        data_path = os.path.join(args.data_path, m, args.penalty, t, 'results.json')
        gather_raw(args, m, t, 'Original', data_path)

        # alternative templates
        _m = m+'_robust'
        for t_type in range(template_num):
            data_path = os.path.join(args.data_path, _m, 'prompt_%d' %t_type, t, 'results.json')
            gather(args, m, t, 'New%d' %(t_type+1), data_path)


    df_dict['layer_depth'] = []
    for i in range(len(df_dict['layer'])):
        df_dict['layer_depth'].append(df_dict['layer'][i]/layer_num[df_dict['model'][i]])

    for key in df_dict:
        if (key != 'model') and (key != 'target'):
            df_dict[key] = np.array(df_dict[key])

    # data frames
    df = pd.DataFrame.from_dict(df_dict)
    df_wo_0 = df.loc[lambda df: df.layer != 0, :]

    out_path = os.path.join(args.data_path, 'robust')
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    # pearson figure
    fig = sns.lineplot(
        data=df_wo_0, 
        x='layer_depth', 
        y='pearson', 
        hue='template_type', 
    )
    savefig(fig, out_path, 'pearson_trend', args.format)

    # r2 figure
    fig = sns.lineplot(
        data=df_wo_0,
        x='layer_depth', 
        y='r2', 
        hue='template_type', 
    )
    savefig(fig, out_path, 'r2_trend', args.format)

    # accuracy figure
    fig = sns.lineplot(
        data=df_wo_0,
        x='layer_depth', 
        y='accuracy', 
        hue='template_type', 
    )
    savefig(fig, out_path, 'accuracy_trend', args.format)

    # mse figure
    fig = sns.lineplot(
        data=df_wo_0,
        x='layer_depth', 
        y='mse', 
        hue='template_type', 
    )

    savefig(fig, out_path, 'mse_trend', args.format)


def main(args):
    tasks = args.tasks.split(',')
    if ('robust' in tasks):
        draw_robust(args) 


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/figures')
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--penalty', type=str, choices=['ridge', 'lasso', 'none'], default='ridge')
    parser.add_argument('--target', type=str, choices=['a', 'b'], required=True)
    parser.add_argument('--tasks', type=str, default='robust')

    args = parser.parse_args()
    print(args)

    main(args)