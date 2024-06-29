# current_date=$(date +"%m%d-%H%M%S")
# sigmas=(0.02 0.03 0.04 0.05 0.06 0.07 0.08 0.09)
# seeds=(0 1 2 3 4 5 6 7)

# # Assuming the length of log_z_values and seeds are the same
# for i in "${!sigmas[@]}"; do
#   sigma=${sigmas[$i]}
#   seed=${seeds[$i]}
#   CUDA_VISIBLE_DEVICES=$seed python src/train.py \
#     --project "alanine_sigma" \
#     --date $current_date \
#     --sigma $sigma \
#     --wandb &
# done

# wait

current_date=$(date +"%m%d-%H%M%S")
for seed in {0..7}; do
  CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --date $current_date \
    --seed $seed \
    --feat_aug \
    --wandb \
    --save_freq 10 \&
done

wait