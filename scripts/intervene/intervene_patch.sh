python ./src/intervene/intervene_patch.py \
    --data_path ./data/embeddings/Mistral-7B_4d/test_processed.data \
    --probe_path ./model/Mistral-7B_4d \
    --penalty ridge \
    --output_path ./data/intervene/patch/Mistral-7B_4d \
    --model_name mistralai/Mistral-7B-v0.1 \
    --model_path /data/public/models/Mistral-7B-v0.1 \
    --max_new_tokens 15 \
    --num_layers 32 \
    --num_examples 300

# python ./src/intervene/intervene_patch.py \
#     --data_path ./data/embeddings/llama-2-7b_4d/train_processed.data \
#     --probe_path ./model/llama-2-7b_4d \
#     --penalty ridge \
#     --output_path ./data/intervene/llama-2-7b \
#     --model_name meta-llama/Llama-2-7b-hf \
#     --model_path /data/public/models/llama2/Llama-2-7b-hf \
#     --max_new_tokens 30 \
#     --num_layers 32 