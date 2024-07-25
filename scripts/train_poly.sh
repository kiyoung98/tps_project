current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py --project poly --molecule poly --date $current_date --seed $seed --start_state pp2 --end_state pp1 --sigma 0.1 --num_steps 10000 --buffer_size 256 --num_samples 2 --batch_size 2 --trains_per_rollout 100 &
done

wait