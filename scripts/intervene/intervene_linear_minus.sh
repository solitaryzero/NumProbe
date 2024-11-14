python ./src/intervene/intervene_linear.py \
    --data_path ./data/4.jsonl \
    --probe_path ./model/Mistral-7B_4d \
    --penalty ridge \
    --output_path ./data/intervene/linear/Mistral-7B_minus \
    --model_name mistralai/Mistral-7B-v0.1 \
    --model_path /data/public/models/Mistral-7B-v0.1 \
    --max_new_tokens 15 \
    --num_layers 32 \
    --num_examples 1000 \
    --task_type layer \
    --task_param 6 \
    --delta -2.0


python ./src/intervene/intervene_linear.py \
    --data_path ./data/4.jsonl \
    --probe_path ./model/Mistral-7B_4d \
    --penalty ridge \
    --output_path ./data/intervene/linear/Mistral-7B_null_minus \
    --model_name mistralai/Mistral-7B-v0.1 \
    --model_path /data/public/models/Mistral-7B-v0.1 \
    --max_new_tokens 15 \
    --num_layers 32 \
    --num_examples 1000 \
    --task_type layer \
    --task_param 6 \
    --null_intervention \
    --delta -2.0


python ./src/intervene/intervene_linear.py \
    --data_path ./data/4.jsonl \
    --probe_path ./model/Mistral-7B_4d \
    --penalty ridge \
    --output_path ./data/intervene/linear/Mistral-7B_random_minus \
    --model_name mistralai/Mistral-7B-v0.1 \
    --model_path /data/public/models/Mistral-7B-v0.1 \
    --max_new_tokens 15 \
    --num_layers 32 \
    --num_examples 1000 \
    --task_type layer \
    --task_param 6 \
    --random_intervention \
    --delta -2.0