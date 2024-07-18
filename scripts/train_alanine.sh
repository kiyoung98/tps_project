current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py --project alanine_div --date $current_date --seed $seed --wandb --save_freq 10 --trains_per_rollout 1000 &
done

wait