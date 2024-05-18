# alanine potential
current_date=$(date +"%m%d-%H%M%S")
bias_scales=(0.01 0.1 1 10 20 50 100 200)
num_gpus=8
idx=0
for gpu in $(seq 0 $((num_gpus-1))); do
  bias_scale=${bias_scales[$idx]}
  CUDA_VISIBLE_DEVICES=$gpu python src/train.py \
    --project alanine_bias_scale \
    --date $current_date \
    --flexible \
    --bias_scale $bias_scale \
    --num_steps 1000 \
    --save_freq 10 \
    --wandb &
  idx=$((idx+1))
  if [ $idx -eq ${#bias_scales[@]} ]; then
    idx=0
  fi
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