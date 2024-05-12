date=0502-152513

for seed in {0..7}
do
  echo ">>" Evaluating poly for seed $seed
  CUDA_VISIBLE_DEVICES=$seed python src/eval.py \
    --date $date \
    --seed $seed \
    --force \
    --bias_scale 1 \
    --wandb &
done

wait