import os
import argparse
import torch
import json
from tqdm import tqdm
import pickle
import time

import numpy as np
from numpy import linalg as LA
import pandas as pd

import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
sns.set_theme(style='ticks')
sns.set_context(rc={"axes.labelsize":20, "legend.fontsize":16, "legend.title_fontsize":16})
mpl.rcParams['lines.markersize'] = 2


def read_prober(args, penalty, layer, target):
    full_path = os.path.join(args.model_path, penalty, 'layer_%d_prober_for_%s.bin' %(layer, target))
    with open(full_path, 'rb') as fin:
        prober = pickle.load(fin)

    return prober


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


def analyze_all_layers(
    args,
):

    gathered_results = {
        'layer': [],
        'sim': [],
    }

    targets = ['a', 'b']
    for layer in range(1, args.layer_count):
        model_a_coef = read_prober(args, args.penalty, layer, 'a').coef_
        model_b_coef = read_prober(args, args.penalty, layer, 'b').coef_

        norm_a, norm_b = LA.norm(model_a_coef), LA.norm(model_b_coef)

        sim = (model_a_coef @ model_b_coef) / norm_a / norm_b
        sim = sim.item()
        gathered_results['layer'].append(layer)
        gathered_results['sim'].append(sim)


    df = pd.DataFrame.from_dict(gathered_results)
    out_path = args.output_path
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)
    # Sim trends
    fig = sns.lineplot(data=df, x='layer', y='sim')
    savefig(fig, out_path, 'direction_similarity', args.format)


    # save gathered results
    with open(os.path.join(args.output_path, 'results.json'), 'w', encoding='utf-8') as fout:
        json.dump(gathered_results, fout)


def main(args):
    analyze_all_layers(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--model_path', type=str, default='./model')
    parser.add_argument('--output_path', type=str, default='./data/figures')
    parser.add_argument('--split', type=str, choices=['val', 'test'], default='test')
    parser.add_argument('--penalty', type=str, choices=['ridge', 'lasso', 'none'], default='ridge')
    parser.add_argument('--layer_count', type=int, default=33)
    parser.add_argument('--logscale_prediction', action='store_true')

    args = parser.parse_args()
    print(args)
    
    main(args)