seeds=(0 1 2 3 4 5 6 7)
current_date=$(date +"%m%d-%H%M%S")

# Iterate over the learning rates
for seed in "${seeds[@]}"; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project trajectory_balance_potential \
    --date $current_date \
    --seed $seed \
    --flexible \
    --num_steps 1000 \
    --wandb &
done

wait