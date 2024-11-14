python ./src/analysis.py \
    --data_path ./data/embeddings_subtraction/llama-2-7b \
    --format pdf \
    --model_path ./model/llama-2-7b_subtraction \
    --output_path ./data/figures_subtraction/llama-2-7b \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction \
    --target all


python ./src/analysis.py \
    --data_path ./data/embeddings_subtraction/llama-2-13b \
    --format pdf \
    --model_path ./model/llama-2-13b_subtraction \
    --output_path ./data/figures_subtraction/llama-2-13b \
    --split test \
    --penalty ridge \
    --layer_count 40 \
    --logscale_prediction \
    --target all


python ./src/analysis.py \
    --data_path ./data/embeddings_subtraction/Mistral-7B \
    --format pdf \
    --model_path ./model/Mistral-7B_subtraction \
    --output_path ./data/figures_subtraction/Mistral-7B \
    --split test \
    --penalty ridge \
    --layer_count 32 \
    --logscale_prediction \
    --target all