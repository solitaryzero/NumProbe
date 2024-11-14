python ./src/refactor_embeds.py \
    --data_path ./data/embeddings_hard/llama-2-7b \
    --model_path /data/public/models/llama2/Llama-2-7b-hf \
    --out_path ./data/embeddings_hard/llama-2-7b

python ./src/refactor_embeds.py \
    --data_path ./data/embeddings_hard/llama-2-13b \
    --model_path /data/public/models/llama2/Llama-2-13b-hf \
    --out_path ./data/embeddings_hard/llama-2-13b

python ./src/refactor_embeds.py \
    --data_path ./data/embeddings_hard/Mistral-7B \
    --model_path /data/public/models/Mistral-7B-v0.1 \
    --out_path ./data/embeddings_hard/Mistral-7B