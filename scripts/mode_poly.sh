current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/mode.py \
    --project mode \
    --date $current_date \
    --seed $seed \
    --num_steps 1000 \
    --flexible \
    --wandb &
done

wait