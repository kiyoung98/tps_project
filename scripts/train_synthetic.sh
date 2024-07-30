current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/synthetic.py --date $current_date --seed $seed --num_samples 10000 --buffer_size 1 --start_temperature 1200 &
done

wait