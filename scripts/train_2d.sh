current_date=$(date +"%m%d-%H%M%S")
sigma_values=(1 2 3 4 5 6 7 8)
seeds=(0 1 2 3 4 5 6 7)

# Assuming the length of log_z_values and seeds are the same
for i in "${!sigma_values[@]}"; do
  sigma=${sigma_values[$i]}
  seed=${seeds[$i]}
  CUDA_VISIBLE_DEVICES=$seed python src/2d_system.py \
    --project "2d" \
    --date $current_date \
    --seed $seed \
    --wandb &
done

wait