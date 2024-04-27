current_date=$(date +"%m%d-%H%M%S")

# Define an array with different learning rates
seeds=(0 1 2 3)

# Initialize a counter variable
gpu=4

# Iterate over the learning rates
for seed in "${seeds[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
    --date $current_date \
    --seed $seed \
    --wandb \
    --project alanine_mass &
  
  # Increment the counter variable
  ((gpu++))
done

wait