current_date=$(date +"%m%d-%H%M")

for seed in {0..7}; do
  echo ">>" Training alanine for $seed
  CUDA_VISIBLE_DEVICES=$seed python src/train.py\
    --seed $seed \
    --wandb \
    --project alanine_mass &
  sleep 0.1
done