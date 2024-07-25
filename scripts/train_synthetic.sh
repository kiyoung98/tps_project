current_date=$(date +"%m%d-%H%M%S")
sigma_values=(0.1 0.2 0.4 0.5 0.6 0.8 0.9 1)
seeds=(0 1 2 3 4 5 6 7)

# Assuming the length of log_z_values and seeds are the same
for i in "${!seeds[@]}"; do
  sigma=${sigma_values[$i]}
  seed=${seeds[$i]}
  CUDA_VISIBLE_DEVICES=$seed python src/synthetic.py \
    --project synthetic_pot \
    --date $current_date \
    --sigma $sigma \
    --dist_feat \
    --wandb &
done

wait