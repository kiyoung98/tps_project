current_date="date"
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/eval.py --project chignolin --molecule chignolin --date $current_date --seed $seed --start_state unfolded --end_state folded --wandb --sigma 0.2 --num_steps 10000 --num_samples 2 &
done

wait