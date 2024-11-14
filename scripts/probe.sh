python ./src/prober.py \
    --data_path ./data/embeddings/llama-2-7b \
    --output_path ./model/llama-2-7b \
    --num_layers 32 \
    --penalty ridge \
    --logscale_prediction \
    --alpha 0.1

python ./src/prober.py \
    --data_path ./data/embeddings/llama-2-13b \
    --output_path ./model/llama-2-13b \
    --num_layers 40 \
    --penalty ridge \
    --logscale_prediction \
    --alpha 0.1

python ./src/prober.py \
    --data_path ./data/embeddings/Mistral-7B \
    --output_path ./model/Mistral-7B \
    --num_layers 32 \
    --penalty ridge \
    --logscale_prediction \
    --alpha 0.1