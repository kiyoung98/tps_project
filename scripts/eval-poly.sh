# for seed in 4
# for seed in {0..7}
for seed in 4
do
  echo ">>" Evaluating poly for seed $seed
  CUDA_VISIBLE_DEVICES=$seed  python src/eval.py \
    --project poly \
    --molecule poly \
    --start_state pp2 \
    --end_state pp1 \
    --num_steps 5000 \
    --seed $seed \
    --wandb
    sleep 0.2
done