import os
import argparse
import json
from tqdm import tqdm, trange
import pickle
import random
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, TensorDataset

import numpy as np
import pandas as pd

from utils import read_dataset, read_embeds, save_model, get_optimizer, get_scheduler, setup_logger, cal_mse


class MLPProber(nn.Module):
    def __init__(self, params):
        super(MLPProber, self).__init__()
        self.prober_layer = nn.Sequential(
            nn.Linear(params['hidden_size'], params['prober_dim']),
            nn.ReLU(),
            nn.Linear(params['prober_dim'], 1),
        )

        self.criterion = nn.MSELoss()

        self.params = params
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() and params.get('cuda', False) else "cpu"
        )
        
        if (params['load_model_path'] is not None):
            self.load_model(params['load_model_path'])


    def load_model(self, fname, cpu=False):
        if cpu:
            state_dict = torch.load(fname, map_location=lambda storage, location: "cpu")
        else:
            state_dict = torch.load(fname)
        self.load_state_dict(state_dict)


    def predict(self,
        hidden_states,
    ):
        predictions = self.prober_layer(hidden_states)

        return predictions.squeeze(1)


    def forward(self, 
        hidden_states,
        labels,
    ):  
        predictions = self.prober_layer(hidden_states).squeeze(1)
        loss = self.criterion(predictions, labels)

        return loss, predictions
    

def evaluate(
    model, 
    eval_dataloader, 
    device, 
    logger,
):
    with torch.no_grad():
        model.eval()
        iter_ = tqdm(eval_dataloader, desc="Evaluation")

        results = {}

        accuracies = []
        mses = []
    
        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            hidden_states, labels = batch
                
            prediction = model.predict(hidden_states )

            prediction = prediction.detach().cpu().numpy()
            labels = labels.detach().cpu().numpy()
            accuracy, mse = cal_mse(prediction, labels)

            accuracies.append(accuracy)
            mses.append(mse)

        normalized_accuracy = np.average(accuracies).item()
        normalized_mse = np.average(mse).item()
        logger.info("Accuracy: %.5f" % normalized_accuracy)
        logger.info("Mean Square Error: %.5f" % normalized_mse)
        results["accuracy"] = normalized_accuracy
        results["mse"] = normalized_mse
        return results


def train(params, target, layer, logger, X_train, Y_train, X_val, Y_val):
    output_path = os.path.join(params['output_path'], 'layer_%d_prober_%s' %(layer, target))
    X_train, Y_train, X_val, Y_val = torch.FloatTensor(X_train), torch.FloatTensor(Y_train), torch.FloatTensor(X_val), torch.FloatTensor(Y_val)

    train_tensor_data = TensorDataset(X_train, Y_train)
    if params["shuffle"]:
        train_sampler = RandomSampler(train_tensor_data)
    else:
        train_sampler = SequentialSampler(train_tensor_data)

    train_dataloader = DataLoader(
        train_tensor_data, sampler=train_sampler, batch_size=params['train_batch_size']
    )

    valid_tensor_data = TensorDataset(X_val, Y_val)
    valid_sampler = SequentialSampler(valid_tensor_data)
    valid_dataloader = DataLoader(
        valid_tensor_data, sampler=valid_sampler, batch_size=params['eval_batch_size']
    )

    model = MLPProber(params)
    model.to(model.device)
    device = model.device

    optimizer = get_optimizer(model, params)
    scheduler = get_scheduler(params, optimizer, len(train_tensor_data), logger)

    best_epoch_idx = -1
    best_score = 1e9
    num_train_epochs = params["epoch"]
    for epoch_idx in trange(int(num_train_epochs), desc="Epoch"):
        model.train()
        tr_loss = 0
        results = None
        iter_ = tqdm(train_dataloader, desc="Batch")

        for step, batch in enumerate(iter_):
            batch = tuple(t.to(device) for t in batch)
            hidden_states, labels = batch
                
            loss, predictions = model(
                hidden_states,
                labels,
            )

            tr_loss += loss.item()

            if (step + 1) % (params["print_interval"]) == 0:
                if (logger):
                    logger.info(
                        "Step %d - epoch %d average loss: %.4f; loss: %.4f" %(
                            step,
                            epoch_idx,
                            tr_loss / (params["print_interval"]),
                            loss.item(),
                        )
                    )
                tr_loss = 0
                # print(predictions)

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(), params["max_grad_norm"]
            )
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        if (logger):
            logger.info("***** Saving fine - tuned model *****")

        epoch_output_folder_path = os.path.join(
            output_path, "epoch_%d" %(epoch_idx)
        )
        save_model(model, epoch_output_folder_path)

        # evaluate
        output_eval_file = os.path.join(epoch_output_folder_path, "eval_results.txt")
        results = evaluate(
            model, valid_dataloader, device=device, logger=logger
        )
        with open(output_eval_file, 'w', encoding='utf-8') as fout:
            fout.write(json.dumps(results, indent=4))

        ls = [best_score, results["mse"]]
        li = [best_epoch_idx, epoch_idx]

        best_score = ls[np.argmin(ls)]
        best_epoch_idx = li[np.argmin(ls)]
        if (logger):
            logger.info("\n")

    # save the best model in the parent_dir
    logger.info("Best performance in epoch: {}".format(best_epoch_idx))
    params["load_model_path"] = os.path.join(
        output_path, 
        "epoch_%d" %(best_epoch_idx),
        'checkpoint.bin',
    )

    model = MLPProber(params)
    model.to(model.device)
    save_model(model, output_path)

    return model
                

def main(args):
    params = args.__dict__
    full_dataset, num_examples = read_dataset(args)

    # Fix the random seeds
    seed = params["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    logger = setup_logger('Prober_mlp', params['output_path'])

    # training
    train_dataset = full_dataset['train']
    num_train_examples = num_examples['train']
    num_layers = args.num_layers

    models = {
        'a': [],
        'b': [],
        # 'predicted': [],
    }

    # score on validation set
    valid_dataset = full_dataset['val']
    num_valid_examples = num_examples['val']

    pos_map = {
        'a': 1, # num1 end
        'b': 3, # num2 end
        # 'predicted': 4, # last token
    }

    for i in tqdm(range(num_layers)):
        all_X_train = read_embeds(args, num_train_examples, 'train', i)
        all_X_val = read_embeds(args, num_valid_examples, 'val', i)

        for target in models:
            X_train = []
            for j, X in enumerate(all_X_train):
                special_positions = train_dataset['special_positions'][j]
                X_train.append(X[special_positions[pos_map[target]], :])
            X_train = np.stack(X_train)

            X_val = []
            for j, X in enumerate(all_X_val):
                special_positions = valid_dataset['special_positions'][j]
                X_val.append(X[special_positions[pos_map[target]], :])
            X_val = np.stack(X_val)

            if (args.logscale_prediction):
                Y_train = np.log2(train_dataset[target])
                Y_val = np.log2(valid_dataset[target])
            else:
                Y_train = train_dataset[target]
                Y_val = valid_dataset[target]
            
            model = train(params, target, i, logger, X_train, Y_train, X_val, Y_val)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--output_path', type=str, default='./model')
    parser.add_argument('--load_model_path', type=str, default=None)
    parser.add_argument('--num_layers', type=int, default=33)

    # model arguments
    parser.add_argument('--hidden_size', type=int, default=4096)
    parser.add_argument('--prober_dim', type=int, default=256)
    parser.add_argument('--cuda', action='store_true')

    # training arguments
    parser.add_argument('--learning_rate', type=float, default=1e-3)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)
    parser.add_argument('--epoch', type=int, default=4)
    parser.add_argument('--warmup_proportion', type=float, default=0.1)
    parser.add_argument('--train_batch_size', type=int, default=32)
    parser.add_argument('--eval_batch_size', type=int, default=64)
    parser.add_argument('--print_interval', type=int, default=200)
    parser.add_argument('--shuffle', action='store_true')
    parser.add_argument('--logscale_prediction', action='store_true')
    parser.add_argument('--seed', type=int, default=42)
    
    args = parser.parse_args()
    print(args)

    main(args)