current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project cysteine \
    --molecule cysteine \
    --date $current_date \
    --seed $seed \
    --wandb \
    --flexible \
    --save_freq 10 \
    --target_std 0.1 \
    --num_steps 1000 \
    --buffer_size 1024 \
    --trains_per_rollout 1000 &
done

wait


# # alanine potential with flexible length
# current_date=$(date +"%m%d-%H%M%S")
# for seed in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$seed python src/train.py \
#     --project alanine_trajectory_balance_potential \
#     --date $current_date \
#     --seed $seed \
#     --wandb \
#     --flexible \
#     --num_steps 1000 &
# done

# wait


# # alanine force
# current_date=$(date +"%m%d-%H%M%S")
# for seed in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$seed python src/train.py \
#     --project alanine_trajectory_balance_force \
#     --date $current_date \
#     --seed $seed \
#     --wandb \
#     --force \
#     --bias_scale 1 &
# done

# wait


# # alanine force with flexible length
# current_date=$(date +"%m%d-%H%M%S")
# for seed in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$seed python src/train.py \
#     --project alanine_trajectory_balance_force \
#     --date $current_date \
#     --seed $seed \
#     --wandb \
#     --force \
#     --bias_scale 1 \
#     --flexible \
#     --num_steps 1000 &
# done

# wait