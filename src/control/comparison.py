import os
import json
import pandas as pd

if __name__ == '__main__':
    example_layers = [2, 10, 20, 30]
    models = ['llama-2-7b', 'llama-2-13b']
    file_paths = {
        'a': './data/figures/%s/ridge/a/results.json',
        'b': './data/figures/%s/ridge/a/results.json',
        'prediction': './data/figures/%s/ridge/predicted_answer/results.json',
        'control': './data/figures/%s_control/ridge/control_signal/results.json',
    }

    for model in models:
        data_dict = {'target': []}
        for key in file_paths:
            path = file_paths[key] %model
            with open(path, 'r', encoding='utf-8') as fin:
                js = json.load(fin)
                l = None
                for feature in js:
                    if (feature not in data_dict):
                        data_dict[feature] = []

                    l = len(js[feature])
                    for f in js[feature]:
                        data_dict[feature].append(f)

                for i in range(l):
                    data_dict['target'].append(key)

        df = pd.DataFrame.from_dict(data_dict)
        bool_list = [x in example_layers for x in df.layer]
        df = df.loc[bool_list, :]
        print(df[['target', 'layer', 'pearson']])
        print(df[['target', 'layer', 'mse']])

        input()