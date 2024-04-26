current_date=$(date +"%m%d-%H%M%S")

echo ">>" Unit test

python src/train.py\
  --project unit_test \
  --date $current_date \
  --num_samples 2

# echo ">>" Evaluating poly for seed $seed
# CUDA_VISIBLE_DEVICES=$seed  python src/eval.py \
#   --molecule alanine \
#   --project unit_test \
#   --date $current_date \
#   --seed $seed \
#   --num_steps 10 \
#   --wandb 
# sleep 0.2

echo ">>" Unit test Done!!

wait