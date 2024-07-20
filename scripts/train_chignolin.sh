current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py --project chignolin --molecule chignolin --date $current_date --seed $seed --start_state unfolded --end_state folded --wandb --save_freq 10 --sigma 0.2 --num_steps 10000 --buffer_size 256 --num_samples 2 --batch_size 2 --trains_per_rollout 100 &
done

wait