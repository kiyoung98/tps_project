current_date=$(date +"%m%d-%H%M%S")

# Define an array with different learning rates
seeds=(0 1 2 3 4 5 6 7)


# Iterate over the learning rates
for seed in "${seeds[@]}"; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --date $current_date \
    --seed $seed \
    --wandb \
    --project alanine_final &

  # Increment the counter variable
done

wait