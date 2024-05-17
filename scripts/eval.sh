project=diverse
date=pot_flex

# for seed in {0..1}
# for seed in {0..7}
for seed in 2
do
  echo ">>" Evaluating poly for seed $seed
  CUDA_VISIBLE_DEVICES=$seed python src/eval.py \
    --date $date \
    --seed $seed \
    --num_samples 64 \
    --num_steps 500 \
    --bias_scale 0.01 \
    --flexible \
    --project $project \
    --wandb &
    sleep 1
done

wait