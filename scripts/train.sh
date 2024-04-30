current_date=$(date +"%m%d-%H%M%S")

# Define an array with different learning rates
trains_per_rollouts=(1 10 100 1000)

# Initialize a counter variable
gpu=4

# Iterate over the learning rates
for trains_per_rollout in "${trains_per_rollouts[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
    --date $current_date \
    --trains_per_rollout $trains_per_rollout \
    --wandb \
    --init_buffer \
    --project alanine_tpr &
  
  # Increment the counter variable
  ((gpu++))
done

wait