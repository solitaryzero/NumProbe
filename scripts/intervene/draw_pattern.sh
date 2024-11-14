python ./src/draw_pattern.py \
    --data_path ./data/embeddings/llama-2-7b_4d \
    --format pdf \
    --model_path ./model/llama-2-7b_4d \
    --output_path ./data/figures/llama-2-7b_4d \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction \
    --target all


python ./src/draw_pattern.py \
    --data_path ./data/embeddings/llama-2-13b_4d \
    --format pdf \
    --model_path ./model/llama-2-13b_4d \
    --output_path ./data/figures/llama-2-13b_4d \
    --split test \
    --penalty ridge \
    --layer_count 40 \
    --logscale_prediction \
    --target all


python ./src/draw_pattern.py \
    --data_path ./data/embeddings/Mistral-7B_4d \
    --format pdf \
    --model_path ./model/Mistral-7B_4d \
    --output_path ./data/figures/Mistral-7B_4d \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction \
    --target all