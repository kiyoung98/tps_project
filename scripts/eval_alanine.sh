current_date="date"
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/eval.py --date $current_date --seed $seed --wandb &
done

wait