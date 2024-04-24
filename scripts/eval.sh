CUDA_VISIBLE_DEVICES=$1  python src/eval.py \
  --project poly_term_std_0.2 \
  --molecule poly \
  --start_state pp2 \
  --end_state pp1 \
  --seed $2