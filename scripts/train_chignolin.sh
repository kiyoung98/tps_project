# current_date=$(date +"%m%d-%H%M%S")
# for seed in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$seed python src/train.py --project chignolin --molecule chignolin --date $current_date --seed $seed --start_state unfolded --end_state folded --wandb --save_freq 10 --sigma 0.2 --num_steps 10000 --buffer_size 256 --num_samples 2 --batch_size 2 --trains_per_rollout 100 &
# done

# wait

current_date=$(date +"%m%d-%H%M%S")
sleep 1
CUDA_VISIBLE_DEVICES=0 python src/train.py --project chignolin_log_z --molecule chignolin --date $current_date --start_state unfolded --end_state folded --wandb --log_z "-0" --save_freq 10 --sigma 0.2 --num_steps 10000 --buffer_size 256 --num_samples 2 --batch_size 2 --trains_per_rollout 100 &
current_date=$(date +"%m%d-%H%M%S")
sleep 1
CUDA_VISIBLE_DEVICES=1 python src/train.py --project chignolin_log_z --molecule chignolin --date $current_date --start_state unfolded --end_state folded --wandb --log_z "-2" --save_freq 10 --sigma 0.2 --num_steps 10000 --buffer_size 256 --num_samples 2 --batch_size 2 --trains_per_rollout 100 &
current_date=$(date +"%m%d-%H%M%S")
sleep 1
CUDA_VISIBLE_DEVICES=2 python src/train.py --project chignolin_log_z --molecule chignolin --date $current_date --start_state unfolded --end_state folded --wandb --log_z "-4" --save_freq 10 --sigma 0.2 --num_steps 10000 --buffer_size 256 --num_samples 2 --batch_size 2 --trains_per_rollout 100 &
current_date=$(date +"%m%d-%H%M%S")
sleep 1
CUDA_VISIBLE_DEVICES=3 python src/train.py --project chignolin_log_z --molecule chignolin --date $current_date --start_state unfolded --end_state folded --wandb --log_z "-6" --save_freq 10 --sigma 0.2 --num_steps 10000 --buffer_size 256 --num_samples 2 --batch_size 2 --trains_per_rollout 100 &
current_date=$(date +"%m%d-%H%M%S")
sleep 1
CUDA_VISIBLE_DEVICES=4 python src/train.py --project chignolin_log_z --molecule chignolin --date $current_date --start_state unfolded --end_state folded --wandb --log_z "-8" --save_freq 10 --sigma 0.2 --num_steps 10000 --buffer_size 256 --num_samples 2 --batch_size 2 --trains_per_rollout 100 &
current_date=$(date +"%m%d-%H%M%S")
sleep 1
CUDA_VISIBLE_DEVICES=5 python src/train.py --project chignolin_log_z --molecule chignolin --date $current_date --start_state unfolded --end_state folded --wandb --log_z "-10" --save_freq 10 --sigma 0.2 --num_steps 10000 --buffer_size 256 --num_samples 2 --batch_size 2 --trains_per_rollout 100 &
current_date=$(date +"%m%d-%H%M%S")
sleep 1
CUDA_VISIBLE_DEVICES=6 python src/train.py --project chignolin_log_z --molecule chignolin --date $current_date --start_state unfolded --end_state folded --wandb --log_z "-12" --save_freq 10 --sigma 0.2 --num_steps 10000 --buffer_size 256 --num_samples 2 --batch_size 2 --trains_per_rollout 100 &
current_date=$(date +"%m%d-%H%M%S")
sleep 1
CUDA_VISIBLE_DEVICES=7 python src/train.py --project chignolin_log_z --molecule chignolin --date $current_date --start_state unfolded --end_state folded --wandb --log_z "-14" --save_freq 10 --sigma 0.2 --num_steps 10000 --buffer_size 256 --num_samples 2 --batch_size 2 --trains_per_rollout 100

