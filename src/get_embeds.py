import os
import json
import argparse
from tqdm import tqdm
import numpy as np

from transformers import AutoModelForCausalLM, AutoTokenizer, GenerationConfig
from datasets import load_dataset, Dataset, DatasetDict


def read_dataset(path):
    template = 'Question: What is the sum of {a} and {b} ?\nAnswer: '
    full_dataset = {
        'train': {'id': [], 'a': [], 'b': [], 'digit': [], 'golden': [], 'prompt': []},
        'val': {'id': [], 'a': [], 'b': [], 'digit': [], 'golden': [], 'prompt': []},
        'test': {'id': [], 'a': [], 'b': [], 'digit': [], 'golden': [], 'prompt': []},
    }
    with open(path, 'r', encoding='utf-8') as fin:
        for line in fin:
            js = json.loads(line)
            prompt = template.format(a=js['a'], b=js['b'])
            spl = js['split']
            full_dataset[spl]['id'].append(js['id'])
            full_dataset[spl]['a'].append(js['a'])
            full_dataset[spl]['b'].append(js['b'])
            full_dataset[spl]['digit'].append(js['num_digits'])
            full_dataset[spl]['golden'].append(js['golden'])
            full_dataset[spl]['prompt'].append(prompt)

    for spl in full_dataset:
        d = Dataset.from_dict(full_dataset[spl])
        full_dataset[spl] = d

    return full_dataset


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        device_map="auto",
    )
    generation_config = GenerationConfig(
        num_beams=1,
        do_sample=False,
        pad_token_id=0,
    )

    gathered_data = {
        'train': [],
        'val': [],
        'test': [],
    }

    if not(os.path.exists(args.output_path)):
        os.makedirs(args.output_path)

    dataset = read_dataset(args.data_path)
    for spl in dataset:
        examples = dataset[spl]
        embed_fouts = []
        if (args.save_embeds):
            for i in range(args.num_layers+1): # including raw embedding layer
                fout = open(os.path.join(args.output_path, '%s_layer_%d.embeds' %(spl, i)), 'wb')
                embed_fouts.append(fout)

        for example in tqdm(examples):
            prompt = example['prompt']
            inputs = tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].cuda()
            generation_outputs = model.generate(
                input_ids=input_ids,
                generation_config=generation_config,
                return_dict_in_generate=True,
                output_hidden_states=args.save_embeds,
                max_new_tokens=args.max_new_tokens,
            )

            tokens = tokenizer.convert_ids_to_tokens(inputs["input_ids"][0])
            l = len(tokens)
            special_positions = [] # start/end of num1, start/end of num2, end of prompt
            num_tokens = '0123456789'
            for i in range(l-1):
                if (tokens[i] in num_tokens) and ((i == 0) or (tokens[i-1] not in num_tokens)):
                    special_positions.append(i)

                if (tokens[i] in num_tokens) and (tokens[i+1] not in num_tokens):
                    special_positions.append(i)

            special_positions.append(l-1)
            assert len(special_positions) == 5

            # hidden states: tuple(tuple(tensor))
            # Shape: max_new_tokens * num_layers * [(batch_size*beams) * step_generation_length * dim]
            # step_generation_length in pos (0, *) = len(input_ids)
            # otherwise step_generation_length = 1 
            answer_start_pos = input_ids.shape[1]
            decoded = tokenizer.decode(generation_outputs.sequences[0], skip_special_tokens=True)

            if (args.save_embeds):
                try:
                    t = generation_outputs.hidden_states[0]
                    for layer_index, tensor in enumerate(t):
                        _tensor = tensor[0, :, :].detach().cpu().numpy()
                        np.save(embed_fouts[layer_index], _tensor)
                except:
                    print(prompt)
                    print(input_ids.shape)
                    print(decoded)
                    print(len(generation_outputs.sequences[0]), answer_start_pos)
                    print(type(generation_outputs.hidden_states), type(generation_outputs.hidden_states[0]), type(generation_outputs.hidden_states[0][0]))
                    print(len(generation_outputs.hidden_states), len(generation_outputs.hidden_states[0]))
                    print(generation_outputs.hidden_states[0][0].shape)
                    quit()
            
            js = example
            js['prediction'] = decoded.strip()
            js['special_positions'] = special_positions
            gathered_data[spl].append(js)

        if (args.save_embeds):
            for fout in embed_fouts:
                fout.close()

        out_path = os.path.join(args.output_path, '%s.data' %spl)
        with open(out_path, 'w', encoding='utf-8') as fout:
            for js in gathered_data[spl]:
                json.dump(js, fout)
                fout.write('\n')



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/raw.jsonl')
    parser.add_argument('--output_path', type=str, default='./data/embeddings')
    parser.add_argument('--model_path', type=str, default='/data/public/models/llama2/Llama-2-7b-hf')
    parser.add_argument('--max_new_tokens', type=int, default=30)
    parser.add_argument('--num_layers', type=int, default=32)
    parser.add_argument('--save_embeds', action='store_true')

    args = parser.parse_args()
    print(args)
    main(args)