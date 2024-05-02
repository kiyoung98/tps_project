train_date=0427-0630

# for seed in 4
for seed in {0..7}
do
  echo ">>" Evaluating poly for seed $seed
  CUDA_VISIBLE_DEVICES=$seed  python src/eval.py \
    --molecule poly \
    --project poly \
    --date $train_date \
    --seed $seed \
    --start_state pp2 \
    --end_state pp1 \
    --num_steps 5000 \
    --wandb &
  sleep 0.5
done