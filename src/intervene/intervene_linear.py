import os
import argparse
import torch
import json
from tqdm import tqdm
import pickle
import time
import random

import numpy as np
from numpy import linalg as LA
import pandas as pd

from transformers import AutoModelForCausalLM, LlamaTokenizer, GenerationConfig
from datasets import load_dataset, Dataset, DatasetDict
import transformer_lens
import transformer_lens.patching as patching

random.seed(42)

def read_dataset(args, path):
    template = 'Question: What is the sum of {a} and {b} ?\nAnswer: '
    full_dataset = {
        'id': [],
        'a': [], 
        'b': [], 
        # 'digit': [], 
        'golden': [], 
        'prompt': [],
    }

    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            prompt = template.format(a=js['a'], b=js['b'])
            full_dataset['id'].append(js['id'])
            full_dataset['a'].append(js['a'])
            full_dataset['b'].append(js['b'])
            full_dataset['golden'].append(js['golden'])
            full_dataset['prompt'].append(prompt)

            if (args.num_examples != -1) and (len(full_dataset['id']) >= args.num_examples):
                break

    d = Dataset.from_dict(full_dataset)
    full_dataset = d

    return full_dataset


def read_prober(args, penalty, layer, target):
    full_path = os.path.join(args.probe_path, penalty, 'layer_%d_prober_for_%s.bin' %(layer, target))
    with open(full_path, 'rb') as fin:
        prober = pickle.load(fin)

    return prober


# find a vector x that is orthogonal to v2 while maximizes cos_sim(v1, x) 
def find_orthogonal_vector(v1, v2):
    # assume v1 = v3 + v4, where v3 // v2 and v4 ⊥ v2
    v3 = v2 * ((v1 @ v2) / (LA.norm(v2) * LA.norm(v2)))
    v4 = v1 - v3
    v4 = v4 / LA.norm(v4)

    return v4


def intervene(hidden, p1, p2, delta):
    orth = find_orthogonal_vector(p1.coef_, p2.coef_) # orth ⊥ p2, |orth| = 1

    delta_orth = orth * (delta/(p1.coef_ @ orth))

    # hidden_pre = hidden[0, -1].detach().cpu().numpy()
    # print('Log2(a) before: ', p1.coef_ @ hidden_pre + p1.intercept_)
    # print('Log2(b) before: ', p2.coef_ @ hidden_pre + p2.intercept_)
    hidden[0, -1] += torch.tensor(delta_orth, device=hidden.device)
    # hidden_post = hidden[0, -1].detach().cpu().numpy()
    # print('Log2(a) after: ', p1.coef_ @ hidden_post + p1.intercept_)
    # print('Log2(b) after: ', p2.coef_ @ hidden_post + p2.intercept_)

    return hidden


def intervene_p(hidden, p, intervene_pos, delta, null=False, random=False):
    if (null):
        v = hidden[0, intervene_pos, :].detach().cpu().numpy()
        v = v / LA.norm(v)
    elif (random):
        v = np.random.rand(p.coef_.shape[0])
        v = v / LA.norm(v)
    else:
        v = p.coef_ / LA.norm(p.coef_)
    # v = p.coef_ / (LA.norm(p.coef_) * LA.norm(p.coef_))
    # v = v / LA.norm(p.coef_)
    delta_p = v*delta

    hidden[:, intervene_pos, :] += torch.tensor(delta_p, device=hidden.device)
    return hidden


def get_answer(decoded_text):
    lines = decoded_text.split('\n')
    for line in lines:
        if (line.startswith('Answer:')):
            if ('is the sum of' in line):
                answer = line.split('is the sum of')[0]
                answer = answer.split('Answer:')[-1].strip()
            elif ('add up to' in line):
                answer = line.split('add up to')[1].split('+')[0]
            elif ('is equal to' in line):
                answer = line.split('is equal to')[1].split('+')[0]
            elif ('is' in line):
                answer = line.split('is')[1].split('+')[0]
            elif ('=' in line):
                answer = line.split('=')[1].split('+')[0].strip()
            else:
                answer = line.split(': ')[-1]
            break

    answer = answer.strip().strip('.').strip()
    answer = answer.replace(',', '')
    answer = answer.replace(' ', '')
    if not(answer.isdigit()) or (int(answer) > 2**62):
        return 0
    else:
        return int(answer)


probes = {}


def main(args):
    # load tokenizer & model
    tokenizer = LlamaTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
    )
    hooked_model = transformer_lens.HookedTransformer.from_pretrained(
        args.model_name,
        fold_ln=False,
        center_writing_weights=False,
        fold_value_biases=False,
        tokenizer=tokenizer,
        hf_model=model,
        device='cuda',
        n_devices=4,
    )
    generation_config = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=0,
    )

    # load probes
    global probes
    probes = {
        'a': [],
        'b': [],
        'p': [],
    }
    for layer in range(args.num_layers):
        probe_a = read_prober(args, args.penalty, layer, 'a')
        probe_b = read_prober(args, args.penalty, layer, 'b')
        probe_p = read_prober(args, args.penalty, layer, 'predicted')
        probes['a'].append(probe_a)
        probes['b'].append(probe_b)
        probes['p'].append(probe_p)

    if not(os.path.exists(args.output_path)):
        os.makedirs(args.output_path)

    dataset = read_dataset(args, args.data_path)
    intervene_pos = -1

    accuracys = {}
    if (args.task_type == 'start'):
        for i in range(args.num_layers-args.task_param+1):
            accuracy = run_intervened(
                args,
                dataset=dataset,
                model=model,
                hooked_model=hooked_model,
                tokenizer=tokenizer,
                probes=probes,
                generation_config=generation_config,
                start_layer=args.task_param,
                end_layer=args.task_param+i,
                intervene_pos=intervene_pos,
            )
            accuracys[i] = accuracy

    elif (args.task_type == 'layer'):
        for i in range(args.num_layers-args.task_param+1):
            accuracy = run_intervened(
                args,
                dataset=dataset,
                model=model,
                hooked_model=hooked_model,
                tokenizer=tokenizer,
                probes=probes,
                generation_config=generation_config,
                start_layer=i,
                end_layer=i+args.task_param-1,
                intervene_pos=intervene_pos,
            )
            accuracys[i] = accuracy

    with open(os.path.join(args.output_path, '%d_%s_int_res.json' %(args.task_param, args.task_type)), 'w', encoding='utf-8') as fout:
        json.dump(accuracys, fout)


def run_intervened(
    args,
    dataset,
    model,
    hooked_model,
    tokenizer,
    probes,
    generation_config,
    start_layer,
    end_layer,
    intervene_pos=-1,
):

    fwd_hooks = []
    gathered_data = []
    total, correct = 0, 0
    
    if (args.null_intervention):
        for layer in range(start_layer, end_layer+1):
            probe_p = probes['p'][layer][intervene_pos]
            fwd_hooks.append((f"blocks.{layer}.hook_resid_post",
                                lambda resid, hook: intervene_p(resid, probe_p, intervene_pos, args.delta, null=True)))
    elif (args.random_intervention):
        for layer in range(start_layer, end_layer+1):
            probe_p = probes['p'][layer][intervene_pos]
            fwd_hooks.append((f"blocks.{layer}.hook_resid_post",
                                lambda resid, hook: intervene_p(resid, probe_p, intervene_pos, args.delta, random=True)))
    else:
        for layer in range(start_layer, end_layer+1):
            probe_p = probes['p'][layer][intervene_pos]
            fwd_hooks.append((f"blocks.{layer}.hook_resid_post",
                                lambda resid, hook: intervene_p(resid, probe_p, intervene_pos, args.delta)))

    for example in tqdm(dataset):
        prompt = example['prompt']
        inputs = tokenizer(prompt, return_tensors="pt")
        input_ids = inputs["input_ids"].cuda()

        # clean run
        generation_outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            max_new_tokens=args.max_new_tokens,
        )
        decoded = tokenizer.decode(generation_outputs.sequences[0], skip_special_tokens=True)

        clean_result = get_answer(decoded)

        # intervened run
        with hooked_model.hooks(fwd_hooks=fwd_hooks):
            hooked_outputs = hooked_model.generate(
                input=input_ids,
                do_sample=False,
                use_past_kv_cache=True,
                max_new_tokens=args.max_new_tokens,
                verbose=False,
            )
            hooked_decoded = tokenizer.decode(hooked_outputs[0], skip_special_tokens=True)
            corrupted_result = get_answer(hooked_decoded)
        
        if (corrupted_result <= 0) or (corrupted_result >= 100000):
            label = False
        else:
            if (args.delta > 0):
                label = (corrupted_result > clean_result)
            else:
                label = (corrupted_result < clean_result) and (corrupted_result != 0)
        dat = {
            'id': example['id'],
            'clean_text': decoded,
            'clean_result': clean_result,
            'intervened_text': hooked_decoded,
            'intervened_result': corrupted_result,
            'label': label,
        }
        gathered_data.append(dat)

        total += 1
        if (label):
            correct += 1

    print('Accuracy: %d/%d = %.4f\n' %(correct, total, correct/total))
    output_path = os.path.join(args.output_path, 'res_layer_%d_to_%d.txt' %(start_layer, end_layer))
    with open(output_path, 'w', encoding='utf-8') as fout:
        fout.write('Accuracy: %d/%d = %.4f\n' %(correct, total, correct/total))
        json.dump(gathered_data, fout)

    return correct/total


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/raw.jsonl')
    parser.add_argument('--probe_path', type=str, default='./model')
    parser.add_argument('--penalty', type=str, default='ridge')
    parser.add_argument('--output_path', type=str, default='./data/intervene')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--model_path', type=str, default='/data/public/models/llama2/Llama-2-7b-hf')
    parser.add_argument('--max_new_tokens', type=int, default=50)
    parser.add_argument('--num_layers', type=int, default=32)

    parser.add_argument('--delta', type=float, default=1.0)
    parser.add_argument('--task_type', type=str)
    parser.add_argument('--task_param', type=int)
    parser.add_argument('--null_intervention', action='store_true')
    parser.add_argument('--random_intervention', action='store_true')
    parser.add_argument('--num_examples', type=int, default=-1)

    args = parser.parse_args()
    print(args)
    main(args)