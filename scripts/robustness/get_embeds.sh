python ./src/robustness/get_embeds.py \
    --data_path ./data/raw.jsonl \
    --output_path ./data/embeddings/llama-2-7b_robust \
    --model_path /data/public/models/llama2/Llama-2-7b-hf \
    --max_new_tokens 50 \
    --num_layers 32 \
    --save_embeds 