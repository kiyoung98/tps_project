current_date="date"
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/eval.py --project poly --molecule poly --date $current_date --seed $seed --start_state pp2 --end_state pp1 --wandb --sigma 0.1 --num_steps 10000 --num_samples 2 &
done

wait