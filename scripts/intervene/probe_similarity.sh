# python ./src/intervene/probe_similarity.py \
#     --format pdf \
#     --model_path ./model/llama-2-7b \
#     --output_path ./data/figures/intervene/llama-2-7b \
#     --split test \
#     --penalty ridge \
#     --layer_count 33 \
#     --logscale_prediction 

python ./src/intervene/probe_similarity.py \
    --format pdf \
    --model_path ./model/Mistral-7B \
    --output_path ./data/figures/intervene/Mistral-7B \
    --split test \
    --penalty ridge \
    --layer_count 33 \
    --logscale_prediction 