current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project aspartic \
    --molecule aspartic \
    --date $current_date \
    --seed $seed \
    --wandb \
    --flexible \
    --save_freq 10 \
    --target_std 0.1 \
    --num_steps 2000 \
    --buffer_size 1024 \
    --trains_per_rollout 1000 &
done

wait