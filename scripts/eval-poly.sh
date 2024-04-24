for seed in {0..7}; do
  echo ">>" Evaluating poly for $seed
  CUDA_VISIBLE_DEVICES=$1  python src/eval.py \
    --project poly_evaly \
    --molecule poly \
    --start_state pp2 \
    --end_state pp1 \
    --seed $seed
    sleep 0.2
done