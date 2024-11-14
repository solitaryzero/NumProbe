import argparse
import os
import json
from tqdm import tqdm

from transformers import AutoTokenizer


def refactor_prediction(record):
    lines = record['prediction'].split('\n')
    for line in lines:
        if (line.startswith('Answer:')):
            if ('is the sum of' in line):
                answer = line.split('is the sum of')[0]
                answer = answer.split('Answer:')[-1].strip()
            elif ('add up to' in line):
                answer = line.split('add up to')[1].split('+')[0]
            elif ('is equal to' in line):
                answer = line.split('is equal to')[1].split('+')[0]
            elif ('is the result of' in line):
                answer = line.split('is the result of')[0]
                answer = answer.split('Answer:')[-1].strip()
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
        print(line)
        print(answer)
        record['predicted'] = 0
    else:
        record['predicted'] = int(answer)


def annotate_correctness(record):
    record['label'] = (record['predicted'] == record['golden'])


def record_tokens(tokenizer, record):
    prompt = record['prompt']
    tokens = tokenizer.tokenize(prompt, add_special_tokens=True)
    record['tokens'] = tokens


def main(args):
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    splits = ['train', 'val', 'test']
    for spl in splits:
        data_path = os.path.join(args.data_path, '%s.data' %spl)
        with open(data_path, 'r', encoding='utf-8') as fin:
            all_records = [json.loads(line) for line in fin]
        for record in tqdm(all_records):
            refactor_prediction(record)
            annotate_correctness(record)
            record_tokens(tokenizer, record)

        if not(os.path.exists(args.out_path)):
            os.makedirs(args.out_path)
        out_path = os.path.join(args.out_path, '%s_processed.data' %spl)
        with open(out_path, 'w', encoding='utf-8') as fout:
            for record in all_records:
                json.dump(record, fout, ensure_ascii=False)
                fout.write('\n')
        


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, default='./data/embeddings')
    parser.add_argument('--model_path', type=str, default='/data/public/models/llama2/Llama-2-7b-hf')
    parser.add_argument('--out_path', type=str, default='./data/embeddings')
    args = parser.parse_args()

    print(args)
    main(args)