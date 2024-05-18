for seed in {0..7}
do
  CUDA_VISIBLE_DEVICES=$seed python src/eval.py \
    --date $date \
    --seed $seed \
    --wandb &
done

wait