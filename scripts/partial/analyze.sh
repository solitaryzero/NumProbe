python ./src/partial/analysis.py \
    --data_path ./data/embeddings/llama-2-7b_8d \
    --format pdf \
    --model_path ./model/llama-2-7b_8d \
    --output_path ./data/figures/partial/llama-2-7b_8d \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction \
    --target a


python ./src/partial/analysis.py \
    --data_path ./data/embeddings/llama-2-13b_8d \
    --format pdf \
    --model_path ./model/llama-2-13b_8d \
    --output_path ./data/figures/partial/llama-2-13b_8d \
    --split test \
    --penalty ridge \
    --layer_count 40 \
    --logscale_prediction \
    --target a


python ./src/partial/analysis.py \
    --data_path ./data/embeddings/Mistral-7B_8d \
    --format pdf \
    --model_path ./model/Mistral-7B_8d \
    --output_path ./data/figures/partial/Mistral-7B_8d \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction \
    --target a