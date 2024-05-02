# chignolin potential
current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project chignolin_trajectory_balance_potential \
    --molecule chignolin \
    --start_state unfolded \
    --end_state folded \
    --num_samples 2 \
    --trains_per_rollout 200 \
    --num_steps 5000 \
    --std 0.05 \
    --target_std 1 \
    --date $current_date \
    --seed $seed \
    --wandb \
    --bias_scale 70 &
done

wait


# chignolin potential with flexible length
current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project chignolin_trajectory_balance_potential \
    --molecule chignolin \
    --start_state unfolded \
    --end_state folded \
    --num_samples 2 \
    --trains_per_rollout 200 \
    --num_steps 5000 \
    --std 0.05 \
    --target_std 1 \
    --date $current_date \
    --seed $seed \
    --wandb \
    --bias_scale 70 \
    --flexible &
done

wait


# chignolin force
current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project chignolin_trajectory_balance_potential \
    --molecule chignolin \
    --start_state unfolded \
    --end_state folded \
    --num_samples 2 \
    --trains_per_rollout 200 \
    --num_steps 5000 \
    --std 0.05 \
    --target_std 1 \
    --date $current_date \
    --seed $seed \
    --wandb \
    --force \
    --bias_scale 1 &
done

wait


# chignolin force with flexible length
current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project chignolin_trajectory_balance_potential \
    --molecule chignolin \
    --start_state unfolded \
    --end_state folded \
    --num_samples 2 \
    --trains_per_rollout 200 \
    --num_steps 5000 \
    --std 0.05 \
    --target_std 1 \
    --date $current_date \
    --seed $seed \
    --wandb \
    --force \
    --bias_scale 1 \
    --flexible &
done

wait