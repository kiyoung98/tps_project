CUDA_VISIBLE_DEVICES=0 python src/train.py --seed 0 --wandb --project alanine &
CUDA_VISIBLE_DEVICES=1 python src/train.py --seed 1 --wandb --project alanine &
CUDA_VISIBLE_DEVICES=2 python src/train.py --seed 2 --wandb --project alanine &
CUDA_VISIBLE_DEVICES=3 python src/train.py --seed 3 --wandb --project alanine &
CUDA_VISIBLE_DEVICES=4 python src/train.py --seed 4 --wandb --project alanine &
CUDA_VISIBLE_DEVICES=5 python src/train.py --seed 5 --wandb --project alanine &
CUDA_VISIBLE_DEVICES=6 python src/train.py --seed 6 --wandb --project alanine &
CUDA_VISIBLE_DEVICES=7 python src/train.py --seed 7 --wandb --project alanine 

CUDA_VISIBLE_DEVICES=0 python src/train.py --seed 0 --wandb --molecule chignolin --project chignolin --start_states unfolded --end_states folded --num_samples 2 --trains_per_rollout 200 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py --seed 1 --wandb --molecule chignolin --project chignolin --start_states unfolded --end_states folded --num_samples 2 --trains_per_rollout 200 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 1 &
CUDA_VISIBLE_DEVICES=2 python src/train.py --seed 2 --wandb --molecule chignolin --project chignolin --start_states unfolded --end_states folded --num_samples 2 --trains_per_rollout 200 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 1 &
CUDA_VISIBLE_DEVICES=3 python src/train.py --seed 3 --wandb --molecule chignolin --project chignolin --start_states unfolded --end_states folded --num_samples 2 --trains_per_rollout 200 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 1 &
CUDA_VISIBLE_DEVICES=4 python src/train.py --seed 4 --wandb --molecule chignolin --project chignolin --start_states unfolded --end_states folded --num_samples 2 --trains_per_rollout 200 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 1 &
CUDA_VISIBLE_DEVICES=5 python src/train.py --seed 5 --wandb --molecule chignolin --project chignolin --start_states unfolded --end_states folded --num_samples 2 --trains_per_rollout 200 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 1 &
CUDA_VISIBLE_DEVICES=6 python src/train.py --seed 6 --wandb --molecule chignolin --project chignolin --start_states unfolded --end_states folded --num_samples 2 --trains_per_rollout 200 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 1 &
CUDA_VISIBLE_DEVICES=7 python src/train.py --seed 7 --wandb --molecule chignolin --project chignolin --start_states unfolded --end_states folded --num_samples 2 --trains_per_rollout 200 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 1 

CUDA_VISIBLE_DEVICES=0 python src/train.py --seed 0 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --learning_rate 0.0001 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py --seed 1 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --learning_rate 0.0001 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.1 &
CUDA_VISIBLE_DEVICES=2 python src/train.py --seed 2 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --learning_rate 0.0001 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.1 &
CUDA_VISIBLE_DEVICES=3 python src/train.py --seed 3 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --learning_rate 0.0001 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.1 &
CUDA_VISIBLE_DEVICES=4 python src/train.py --seed 4 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --learning_rate 0.0001 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.1 &
CUDA_VISIBLE_DEVICES=5 python src/train.py --seed 5 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --learning_rate 0.0001 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.1 &
CUDA_VISIBLE_DEVICES=6 python src/train.py --seed 6 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --learning_rate 0.0001 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.1 &
CUDA_VISIBLE_DEVICES=7 python src/train.py --seed 7 --wandb --molecule poly --project poly --start_states pp2 --end_states pp1 --learning_rate 0.0001 --num_samples 2 --trains_per_rollout 1000 --num_steps 5000 --start_std 0.1 --end_std 0.05 --terminal_std 0.1 
