python ./src/analysis_mlp.py \
    --data_path ./data/embeddings/llama-2-7b \
    --model_path ./model/llama-2-7b_mlp \
    --output_path ./data/figures/llama-2-7b_mlp \
    --split test \
    --hidden_size 4096 \
    --prober_dim 256 \
    --logscale_prediction \
    --target a
