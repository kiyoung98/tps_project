project=visual
date=pot

for seed in 0
do
  echo ">>" Evaluating chig for seed $seed
  CUDA_VISIBLE_DEVICES=$1 python src/eval.py \
    --molecule chignolin \
    --project $project \
    --date $date \
    --seed $seed \
    --start_state unfolded \
    --end_state folded \
    --num_samples 32 \
    --num_steps 5000 \
    --bias_scale 0.01 \
    --flexible
    sleep 1
done
