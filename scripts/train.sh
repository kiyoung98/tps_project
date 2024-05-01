current_date=$(date +"%m%d-%H%M%S")

# Define an array with different learning rates
std_scales=(1.5 2 2.5 3 3.5 4 4.5 5)

gpu=0

# Iterate over the learning rates
for std_scale in "${std_scales[@]}"; do
  CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
    --date $current_date \
    --std_scale $std_scale \
    --wandb \
    --flexible \
    --project alanine_std_scale &

  # Increment the counter variable
  ((gpu++))
done

wait