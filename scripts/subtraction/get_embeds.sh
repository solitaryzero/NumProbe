python ./src/subtraction/get_embeds.py \
    --data_path ./data/subtraction.jsonl \
    --output_path ./data/embeddings_subtraction/llama-2-7b \
    --model_path /data/public/models/llama2/Llama-2-7b-hf \
    --max_new_tokens 30 \
    --num_layers 32 \
    --save_embeds 

python ./src/subtraction/get_embeds.py \
    --data_path ./data/subtraction.jsonl \
    --output_path ./data/embeddings_subtraction/llama-2-13b \
    --model_path /data/public/models/llama2/Llama-2-13b-hf \
    --max_new_tokens 30 \
    --num_layers 40 \
    --save_embeds 

python ./src/subtraction/get_embeds.py \
    --data_path ./data/subtraction.jsonl \
    --output_path ./data/embeddings_subtraction/Mistral-7B \
    --model_path /data/public/models/Mistral-7B-v0.1 \
    --max_new_tokens 30 \
    --num_layers 32 \
    --save_embeds