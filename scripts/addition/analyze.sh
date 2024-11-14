# python ./src/addition/analysis_addition.py \
#     --data_path ./data/embeddings_hard/llama-2-7b \
#     --format pdf \
#     --model_path ./model_hard/llama-2-7b \
#     --output_path ./data/figures/addition/llama-2-7b \
#     --split test \
#     --penalty ridge \
#     --layer_count 32 \
#     --logscale_prediction

# python ./src/addition/analysis_addition.py \
#     --data_path ./data/embeddings_hard/llama-2-13b \
#     --format pdf \
#     --model_path ./model_hard/llama-2-13b \
#     --output_path ./data/figures/addition/llama-2-13b \
#     --split test \
#     --penalty ridge \
#     --layer_count 40 \
#     --logscale_prediction

# python ./src/addition/analysis_addition.py \
#     --data_path ./data/embeddings_hard/Mistral-7B \
#     --format pdf \
#     --model_path ./model_hard/Mistral-7B \
#     --output_path ./data/figures/addition/Mistral-7B \
#     --split test \
#     --penalty ridge \
#     --layer_count 32 \
#     --logscale_prediction

python ./src/addition/analysis_addition.py \
    --data_path ./data/embeddings/llama-2-7b \
    --format pdf \
    --model_path ./model/llama-2-7b \
    --output_path ./data/figures/addition/llama-2-7b \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction

python ./src/addition/analysis_addition.py \
    --data_path ./data/embeddings/llama-2-13b \
    --format pdf \
    --model_path ./model/llama-2-13b \
    --output_path ./data/figures/addition/llama-2-13b \
    --split test \
    --penalty ridge \
    --layer_count 40 \
    --logscale_prediction

python ./src/addition/analysis_addition.py \
    --data_path ./data/embeddings/Mistral-7B \
    --format pdf \
    --model_path ./model/Mistral-7B \
    --output_path ./data/figures/addition/Mistral-7B \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction