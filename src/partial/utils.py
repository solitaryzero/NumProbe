import os
import sys
import argparse
import torch
import json
from tqdm import tqdm
import pickle
import logging

import numpy as np
import pandas as pd
from transformers import get_linear_schedule_with_warmup


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

    return embeddings


def read_prober(args, penalty, layer, target):
    full_path = os.path.join(args.model_path, penalty, 'layer_%d_prober.bin' %(layer))
    with open(full_path, 'rb') as fin:
        prober = pickle.load(fin)

    return prober


def setup_logger(name, save_dir, filename="log.txt", level=logging.INFO):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter("%(asctime)s - %(filename)s[line:%(lineno)d] - %(levelname)s: %(message)s")
    
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if save_dir:
        if not(os.path.exists(save_dir)):
            os.makedirs(save_dir)
        fh = logging.FileHandler(os.path.join(save_dir, filename))
        fh.setLevel(level)
        fh.setFormatter(formatter)
        logger.addHandler(fh)

    return logger


def save_model(model, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_model_file = os.path.join(output_dir, 'checkpoint.bin')
    torch.save(model.state_dict(), output_model_file)


def ellipse(lst, max_display=5, sep='|'):
    """
    Like join, but possibly inserts an ellipsis.
    :param lst: The list to join on
    :param int max_display: the number of items to display for ellipsing.
        If -1, shows all items
    :param string sep: the delimiter to join on
    """
    # copy the list (or force it to a list if it's a set)
    choices = list(lst)
    # insert the ellipsis if necessary
    if max_display > 0 and len(choices) > max_display:
        ellipsis = '...and {} more'.format(len(choices) - max_display)
        choices = choices[:max_display] + [ellipsis]
    return sep.join(str(c) for c in choices)


def get_optimizer(model, params):
    parameters_with_decay = []
    parameters_with_decay_names = []
    parameters_without_decay = []
    parameters_without_decay_names = []
    no_decay = ['bias', 'gamma', 'beta']

    for n, p in model.named_parameters():
        if any(t in n for t in no_decay):
            parameters_without_decay.append(p)
            parameters_without_decay_names.append(n)
        else:
            parameters_with_decay.append(p)
            parameters_with_decay_names.append(n)

    print('The following parameters will be optimized WITH decay:')
    print(ellipse(parameters_with_decay_names, 5, ' , '))
    print('The following parameters will be optimized WITHOUT decay:')
    print(ellipse(parameters_without_decay_names, 5, ' , '))

    optimizer_grouped_parameters = [
        {'params': parameters_with_decay, 'weight_decay': 0.01},
        {'params': parameters_without_decay, 'weight_decay': 0.0},
    ]
    optimizer = torch.optim.AdamW(
        optimizer_grouped_parameters, 
        lr=params['learning_rate'],
    )

    return optimizer


def get_scheduler(params, optimizer, len_train_data, logger):
    batch_size = params['train_batch_size']
    epochs = params['epoch']

    num_train_steps = int(len_train_data / batch_size) * epochs
    num_warmup_steps = int(num_train_steps * params['warmup_proportion'])

    scheduler = get_linear_schedule_with_warmup(
        optimizer, 
        num_warmup_steps=num_warmup_steps, 
        num_training_steps=num_train_steps,
    )
    if (logger):
        logger.info("Num optimization steps = %d" % num_train_steps)
        logger.info("Num warmup steps = %d", num_warmup_steps)
    return scheduler


def cal_mse(predictions, labels):
    total_count = labels.shape[0]
    correct_count = np.sum(predictions == labels)
    accuracy = correct_count / total_count

    mse = np.mean((predictions-labels)**2)
    return accuracy, mse