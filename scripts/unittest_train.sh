CUDA_VISIBLE_DEVICES=$1 python src/train.py \
    --terminal_std 0.1 \
    --learning_rate 1e-1 \
    --num_rollouts 4 \
    --buffer_size 4 \
    --num_steps 4 \
    --seed 0