python ./src/get_embeds.py \
    --data_path ./data/8.jsonl \
    --output_path ./data/embeddings/llama-2-7b_8d \
    --model_path /data/public/models/llama2/Llama-2-7b-hf \
    --max_new_tokens 30 \
    --num_layers 32 \
    --save_embeds 

python ./src/get_embeds.py \
    --data_path ./data/8.jsonl \
    --output_path ./data/embeddings/llama-2-13b_8d \
    --model_path /data/public/models/llama2/Llama-2-13b-hf \
    --max_new_tokens 30 \
    --num_layers 40 \
    --save_embeds 

python ./src/get_embeds.py \
    --data_path ./data/8.jsonl \
    --output_path ./data/embeddings/Mistral-7B_8d \
    --model_path /data/public/models/Mistral-7B-v0.1 \
    --max_new_tokens 30 \
    --num_layers 32 \
    --save_embeds 