import os
import argparse
import json
from tqdm import tqdm
import time

import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.ticker import MaxNLocator
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
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    save_path = os.path.join(out_path, '%s.%s' %(file_name, format))
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
    fig.figure.savefig(save_path, format=format)
    close_plots()


def savegrid(grid, out_path, file_name, format):
    if not(os.path.exists(out_path)):
        os.makedirs(out_path)

    save_path = os.path.join(out_path, '%s.%s' %(file_name, format))
    grid.tick_params(labelsize=16)
    grid.fig.tight_layout(w_pad=1)
    grid.figure.savefig(save_path, format=format)
    close_plots()


def draw_linear(args):
    model = 'Mistral-7B'
    targets = {
        '5_layer': ('layer', 5, '5_layer_ref.json'),
        '6_layer': ('layer', 6, '6_layer_ref.json'),
        '14_start': ('start', 14, '14_start_ref.json'),
    }
    in_path = os.path.join(args.data_path, 'linear', model)

    for target in targets:
        task_type, task_param, file_name = targets[target]
        data_path = os.path.join(in_path, file_name)
        with open(data_path, 'r', encoding='utf-8') as fin:
            js = json.load(fin)
            if (task_type == 'layer'):
                df_dict = {
                    'start_layer': [],
                    'success_rate': [],
                }
                for key in js:
                    df_dict['start_layer'].append(int(key))
                    df_dict['success_rate'].append(js[key])

                df = pd.DataFrame.from_dict(df_dict)
                df['start_layer'] = df['start_layer'].astype(int)
                fig = sns.lineplot(
                    data=df, 
                    x='start_layer', 
                    y='success_rate',
                )
                fig.set_xlabel('Start Layer')
                fig.set_ylabel('Success Rate')
                savefig(fig, args.out_path, target, args.format)
            elif (task_type == 'start'):
                df_dict = {
                    'intervened_layers': [],
                    'success_rate': [],
                }
                for key in js:
                    df_dict['intervened_layers'].append(int(key)-task_param+1)
                    df_dict['success_rate'].append(js[key])

                df = pd.DataFrame.from_dict(df_dict)
                df['intervened_layers'] = df['intervened_layers'].astype(int)
                fig = sns.lineplot(
                    data=df, 
                    x='intervened_layers', 
                    y='success_rate',
                )
                fig.xaxis.set_major_locator(MaxNLocator(integer=True))
                fig.set_xlabel('Layers Intervened')
                fig.set_ylabel('Success Rate')
                savefig(fig, args.out_path, target, args.format)


def draw_linear_full_comparision(args, minus):
    if not(minus):
        models = [
            ('Mistral-7B', 'Linear', 'filtered_results.json'), 
            ('Mistral-7B_null', 'Null', 'filtered_results.json'),
            ('Mistral-7B_random', 'Random', 'filtered_results.json'),
        ]
    else:
        models = [
            ('Mistral-7B_minus', 'Linear', 'filtered_results.json'), 
            ('Mistral-7B_null_minus', 'Null', 'filtered_results.json'),
            ('Mistral-7B_random_minus', 'Random', 'filtered_results.json'),
        ]

    df_dict = {
        'start_layer': [],
        'success_rate': [],
        'method': [],
    }
    for model, method, file_name in models:
        in_path = os.path.join(args.data_path, model)

        data_path = os.path.join(in_path, file_name)
        with open(data_path, 'r', encoding='utf-8') as fin:
            js = json.load(fin)
            
            for key in js:
                df_dict['start_layer'].append(int(key))
                df_dict['success_rate'].append(js[key])
                df_dict['method'].append(method)

        df = pd.DataFrame.from_dict(df_dict)
        df['start_layer'] = df['start_layer'].astype(int)

        fig = sns.lineplot(
            data=df, 
            x='start_layer', 
            y='success_rate',
            hue='method',
        )
        fig.set_xlabel('Start Layer')
        fig.set_ylabel('Success Rate')
        if not(minus):
            savefig(fig, args.out_path, 'null_comparison', args.format)
        else:
            savefig(fig, args.out_path, 'null_comparison_minus', args.format)


def draw_patch(args):
    model = 'Mistral-7B_4d'
    data_path = os.path.join(args.data_path, 'patch', model, 'patching_results.json')
    with open(data_path, 'r', encoding='utf-8') as fin:
        js = json.load(fin)

    effects = {}

    for entry in js:
        key = (entry['layer'], entry['position'])
        if (key not in effects):
            effects[key] = []

        effect = (entry['intervened_result']-entry['clean_result']) / (entry['corrupted_result']-entry['clean_result'])
        effects[key].append(effect)

    df_dict = {
        'layer': [],
        'position': [],
        'effect': [],
    }
    for key in effects:
        layer, position = key
        effect = sum(effects[key])/len(effects[key])
        df_dict['layer'].append(layer)
        df_dict['position'].append(position)
        df_dict['effect'].append(effect)

    df = pd.DataFrame.from_dict(df_dict)
    df['layer'] = df['layer'].astype(int)

    converted_df = df.pivot(index='position', columns='layer', values='effect')
    index_order = ['early', '<a0>', '<a1>', '<a2>', '<a3>', 'mid', 'last']
    converted_df = converted_df.reindex(index=index_order)
    fig = sns.heatmap(data=converted_df, annot=False, cmap='Blues')

    fig.tick_params(axis='y', labelrotation=0)
    x_labels = fig.get_xticklabels()
    kept_labels = x_labels[::2]
    x_labels = ['' for x in x_labels]
    x_labels[::2] = kept_labels
    fig.set_xticklabels(x_labels)

    savefig(fig, args.out_path, 'patch_effect', args.format)


def main(args):
    tasks = args.tasks.split(',')
    if ('linear' in tasks):
        draw_linear(args)
    if ('full' in tasks):
        draw_linear_full_comparision(args, minus=True)
        draw_linear_full_comparision(args, minus=False)
    if ('patch' in tasks):
        draw_patch(args)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/figures')
    parser.add_argument('--format', type=str, choices=['png', 'pdf'], default='png')
    parser.add_argument('--out_path', type=str, default='./data/figures')
    parser.add_argument('--tasks', type=str, default='linear')

    args = parser.parse_args()
    print(args)

    main(args)