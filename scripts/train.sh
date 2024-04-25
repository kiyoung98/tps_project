current_date=$(date +"%m%d-%H%M")

for seed in {0..7}; do
  echo ">>" Training alanine for $seed
    CUDA_VISIBLE_DEVICES=$seed python src/train.py \
    --date $current_date \
    --seed 0 \
    --bias_scale $((seed*100+100)) \
    --wandb \
    --num_steps 600 \
    --project alanine_mass &
done

wait