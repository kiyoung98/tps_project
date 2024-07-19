current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py --project alanine --date $current_date --seed $seed --wandb --save_freq 10 &
done

wait