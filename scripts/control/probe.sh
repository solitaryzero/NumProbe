python ./src/control/control_probe.py \
    --data_path ./data/embeddings/llama-2-7b \
    --control_path ./data/control.jsonl \
    --output_path ./model/llama-2-7b_control \
    --num_layers 32 \
    --penalty ridge \
    --logscale_prediction \
    --alpha 0.1

python ./src/control/control_probe.py \
    --data_path ./data/embeddings/llama-2-13b \
    --control_path ./data/control.jsonl \
    --output_path ./model/llama-2-13b_control \
    --num_layers 40 \
    --penalty ridge \
    --logscale_prediction \
    --alpha 0.1

python ./src/control/control_probe.py \
    --data_path ./data/embeddings/Mistral-7B \
    --control_path ./data/control.jsonl \
    --output_path ./model/Mistral-7B_control \
    --num_layers 32 \
    --penalty ridge \
    --logscale_prediction \
    --alpha 0.1