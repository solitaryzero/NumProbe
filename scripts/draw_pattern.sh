python ./src/draw_pattern.py \
    --data_path ./data/embeddings/llama-2-7b \
    --format pdf \
    --model_path ./model/llama-2-7b \
    --output_path ./data/figures/llama-2-7b \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction \
    --target all


python ./src/draw_pattern.py \
    --data_path ./data/embeddings/llama-2-13b \
    --format pdf \
    --model_path ./model/llama-2-13b \
    --output_path ./data/figures/llama-2-13b \
    --split test \
    --penalty ridge \
    --layer_count 40 \
    --logscale_prediction \
    --target all


python ./src/draw_pattern.py \
    --data_path ./data/embeddings/Mistral-7B \
    --format pdf \
    --model_path ./model/Mistral-7B \
    --output_path ./data/figures/Mistral-7B \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction \
    --target all