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
        'golden': [], 
        'prompt': [],
        'special_positions': [],
        'corrupted_a': [],
        'corrupted_golden': [],
        'corrupted_prompt': [],
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
            full_dataset['special_positions'].append(js['special_positions'])

            # corrupted_a = random.randint(1000, 9999)
            corrupted_a = 9999
            corrupted_golden = corrupted_a + js['b']

            full_dataset['corrupted_a'].append(corrupted_a)
            full_dataset['corrupted_golden'].append(corrupted_golden)
            corrupted_prompt = template.format(a=corrupted_a, b=js['b'])
            full_dataset['corrupted_prompt'].append(corrupted_prompt)

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


def intervene(hidden, corrupted_hidden, pos=-1):
    hidden[:, pos] = corrupted_hidden[:, pos]
    return hidden


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

    gathered_data = []

    if not(os.path.exists(args.output_path)):
        os.makedirs(args.output_path)

    dataset = read_dataset(args, args.data_path)

    for example in tqdm(dataset):
        # clean run
        prompt = example['prompt']
        inputs = tokenizer(prompt, return_tensors="pt")
        tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
        input_ids = inputs["input_ids"].cuda()

        generation_outputs = model.generate(
            input_ids=input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            max_new_tokens=args.max_new_tokens,
        )
        decoded = tokenizer.decode(generation_outputs.sequences[0], skip_special_tokens=True)
        print('Clean run:')
        print(decoded)
        clean_result = get_answer(decoded)

        # corrupted cache
        corrupted_prompt = example['corrupted_prompt']
        corrupted_inputs = tokenizer(corrupted_prompt, return_tensors="pt")
        # corrupted_tokens = tokenizer.convert_ids_to_tokens(corrupted_inputs["input_ids"][0])
        corrupted_input_ids = corrupted_inputs["input_ids"].cuda()

        generation_outputs = model.generate(
            input_ids=corrupted_input_ids,
            generation_config=generation_config,
            return_dict_in_generate=True,
            max_new_tokens=args.max_new_tokens,
        )
        decoded = tokenizer.decode(generation_outputs.sequences[0], skip_special_tokens=True)
        print('Corrupted run:')
        print(decoded)
        corrupted_result = get_answer(decoded)

        _logits, corrupted_cache = hooked_model.run_with_cache(
            input=corrupted_input_ids,
        )

        # intervened run
        num1_start, num1_end, num2_start, num2_end, last_token = example['special_positions']
        intervene_positions = [] # early
        intervene_positions.append(num1_start-1) # early
        intervene_positions.extend(list(range(num1_start, num1_end+1))) # input number a
        intervene_positions.append(num1_end+1) # mid
        intervene_positions.append(last_token) # last
        intervene_tokens = ['early', '<a0>', '<a1>', '<a2>', '<a3>', 'mid', 'last']
        assert len(intervene_tokens) == len(intervene_positions)

        for index, intervene_pos in tqdm(enumerate(intervene_positions)):
            for layer in range(args.num_layers):
                def intervene_hook(resid, hook):
                    if (resid.shape[1] == 1):
                        return resid
                
                    return intervene(resid, corrupted_cache[hook.name], intervene_pos)

                with hooked_model.hooks(fwd_hooks=[(f"blocks.{layer}.hook_resid_post", intervene_hook)]):
                    hooked_outputs = hooked_model.generate(
                        input=input_ids,
                        do_sample=False,
                        use_past_kv_cache=True,
                        max_new_tokens=args.max_new_tokens,
                        verbose=False,
                    )
                    hooked_decoded = tokenizer.decode(hooked_outputs[0], skip_special_tokens=True)
                    # print(f'Layer {layer}:')
                    # print(f'Position {intervene_tokens[index]}:')
                    # print(hooked_decoded)
                    # input()
                    intervened_result = get_answer(hooked_decoded)

                    gathered_data.append({
                        'id': example['id'],
                        'a': example['a'],
                        'b': example['b'],
                        'clean_result': clean_result,
                        'corrupted_result': corrupted_result,
                        'intervened_result': intervened_result,
                        'layer': layer,
                        'position': intervene_tokens[index],
                    })

    output_path = os.path.join(args.output_path, 'patching_results.json')
    with open(output_path, 'w', encoding='utf-8') as fout:
        json.dump(gathered_data, fout)
            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/raw.jsonl')
    parser.add_argument('--probe_path', type=str, default='./model')
    parser.add_argument('--penalty', type=str, default='ridge')
    parser.add_argument('--output_path', type=str, default='./data/intervene')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-2-7b-hf')
    parser.add_argument('--model_path', type=str, default='/data/public/models/llama2/Llama-2-7b-hf')
    parser.add_argument('--max_new_tokens', type=int, default=30)
    parser.add_argument('--num_layers', type=int, default=33)

    parser.add_argument('--num_examples', type=int, default=-1)

    args = parser.parse_args()
    print(args)
    main(args)