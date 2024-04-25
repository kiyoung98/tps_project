current_date=$(date +"%m%d-%H%M")

for seed in {0..7}; do
  echo ">>" Training poly for $seed
  CUDA_VISIBLE_DEVICES=$seed python src/train.py\
    --seed $seed \
    --wandb \
    --molecule poly \
    --project $current_date \
    --start_states pp2 \
    --end_states pp1 \
    --num_samples 2 \
    --trains_per_rollout 1000 \
    --num_steps 5000 \
    --start_std 0.1 \
    --end_std 0.05 \
    --learning_rate 0.01 \
    --terminal_std 0.2 &
    sleep 0.2
done

# CUDA_VISIBLE_DEVICES=1 python src/train.py --seed 1 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.5 &
# CUDA_VISIBLE_DEVICES=2 python src/train.py --seed 2 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.5 &
# CUDA_VISIBLE_DEVICES=3 python src/train.py --seed 3 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.5 &
# CUDA_VISIBLE_DEVICES=4 python src/train.py --seed 4 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.5 &
# CUDA_VISIBLE_DEVICES=5 python src/train.py --seed 5 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.5 &
# CUDA_VISIBLE_DEVICES=6 python src/train.py --seed 6 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.5 &
# CUDA_VISIBLE_DEVICES=7 python src/train.py --seed 7 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.5 