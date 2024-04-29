current_date=$(date +"%m%d-%H%M")

for seed in {0..7}; do
  echo ">>" Training poly for $seed
  CUDA_VISIBLE_DEVICES=$seed python src/train.py\
    --molecule poly \
    --project poly \
    --date $current_date \
    --seed $seed \
    --start_states pp2 \
    --end_states pp1 \
    --num_steps 5000 \
    --start_std 0.1 \
    --end_std 0.05 \
    --num_rollouts 4000 \
    --learning_rate 0.008 \
    --terminal_std 0.2 \
    --wandb &
    sleep 0.2
done