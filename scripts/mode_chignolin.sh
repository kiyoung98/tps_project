# current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/mode_chignolin.py \
    --project chignolin_mode \
    --date $seed \
    --target_std $(echo "scale=1; $seed * 0.02 + 0.01" | bc) \
    --flexible \
    --wandb &
done

wait