python ./src/control/analysis.py \
    --data_path ./data/embeddings/llama-2-7b \
    --control_path ./data/control.jsonl \
    --format pdf \
    --model_path ./model/llama-2-7b_control \
    --output_path ./data/figures/control/llama-2-7b \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction 


python ./src/control/analysis.py \
    --data_path ./data/embeddings/llama-2-13b \
    --control_path ./data/control.jsonl \
    --format pdf \
    --model_path ./model/llama-2-13b_control \
    --output_path ./data/figures/control/llama-2-13b \
    --split test \
    --penalty ridge \
    --layer_count 40 \
    --logscale_prediction 

python ./src/control/analysis.py \
    --data_path ./data/embeddings/Mistral-7B \
    --control_path ./data/control.jsonl \
    --format pdf \
    --model_path ./model/Mistral-7B_control \
    --output_path ./data/figures/control/Mistral-7B \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction 