# poly potential
current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project poly_trajectory_balance_potential \
    --molecule poly \
    --start_state pp2 \
    --end_state pp1 \
    --num_samples 2 \
    --trains_per_rollout 1000 \
    --num_steps 5000 \
    --std 0.05 \
    --target_std 0.05 \
    --date $current_date \
    --seed $seed \
    --wandb \
    --bias_scale 70 &
done

wait


# poly potential with flexible length
current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project poly_trajectory_balance_potential \
    --molecule poly \
    --start_state pp2 \
    --end_state pp1 \
    --num_samples 2 \
    --trains_per_rollout 1000 \
    --num_steps 5000 \
    --std 0.05 \
    --target_std 0.05 \
    --date $current_date \
    --seed $seed \
    --wandb \
    --bias_scale 70 \
    --flexible &
done

wait


# poly force
current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project poly_trajectory_balance_potential \
    --molecule poly \
    --start_state pp2 \
    --end_state pp1 \
    --num_samples 2 \
    --trains_per_rollout 1000 \
    --num_steps 5000 \
    --std 0.05 \
    --target_std 0.05 \
    --date $current_date \
    --seed $seed \
    --wandb \
    --force \
    --bias_scale 1 &
done

wait


# poly force with flexible length
current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project poly_trajectory_balance_potential \
    --molecule poly \
    --start_state pp2 \
    --end_state pp1 \
    --num_samples 2 \
    --trains_per_rollout 1000 \
    --num_steps 5000 \
    --std 0.05 \
    --target_std 0.05 \
    --date $current_date \
    --seed $seed \
    --wandb \
    --force \
    --bias_scale 1 \
    --flexible &
done

wait