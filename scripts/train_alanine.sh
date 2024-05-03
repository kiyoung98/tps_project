# # alanine potential
# current_date=$(date +"%m%d-%H%M%S")
# for seed in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$seed python src/train.py \
#     --project alanine \
#     --date $current_date \
#     --seed $seed \
#     --wandb &
# done

# wait


# # alanine potential with flexible length
# current_date=$(date +"%m%d-%H%M%S")
# for seed in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$seed python src/train.py \
#     --project alanine \
#     --date $current_date \
#     --seed $seed \
#     --wandb \
#     --flexible \
#     --num_steps 1000 &
# done

# wait


# alanine force
current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --project alanine \
    --date $current_date \
    --seed $seed \
    --wandb \
    --force \
    --bias_scale 1 &
done

wait


# # alanine force with flexible length
# current_date=$(date +"%m%d-%H%M%S")
# for seed in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$seed python src/train.py \
#     --project alanine \
#     --date $current_date \
#     --seed $seed \
#     --wandb \
#     --force \
#     --bias_scale 1 \
#     --flexible \
#     --num_steps 1000 &
# done

# wait