export TZ=Asia/Seoul
current_date=$(date +"%m%d-%H%M%S")

echo ">>" Unit test for poly

for seed in {0..1}
do
  echo ">>" Training poly for seed $seed
  CUDA_VISIBLE_DEVICES=$seed python src/train.py\
    --molecule alanine \
    --project unit_test \
    --date $current_date \
    --seed $seed \
    --num_samples 4 \
    --trains_per_rollout 100 \
    --num_rollouts 32 \
    --num_steps 100 \
    --start_std 0.1 \
    --end_std 0.05 \
    --learning_rate 0.01 \
    --terminal_std 0.2 \
    --freq_rollout_save 1 \
    --wandb
  sleep 0.2

  echo ">>" Evaluating poly for seed $seed
  CUDA_VISIBLE_DEVICES=$seed  python src/eval.py \
    --molecule alanine \
    --project unit_test \
    --date $current_date \
    --seed $seed \
    --num_steps 10 \
    --wandb 
  sleep 0.2
done
echo ">>" Unit test Done!!