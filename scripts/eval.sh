date=pot_flex

for seed in {0..7}
do
  echo ">>" Evaluating poly for seed $seed
  CUDA_VISIBLE_DEVICES=$seed python src/eval.py \
    --date $date \
    --seed $seed \
    --num_steps 1000 \
    --flexible \
    --project mode \
    --wandb &
done

wait