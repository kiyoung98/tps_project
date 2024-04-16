CUDA_VISIBLE_DEVICES=0 python src/train.py --wandb --z_scale 10 --terminal_std 0.001 --learning_rate 0.1 --force &
CUDA_VISIBLE_DEVICES=1 python src/train.py --wandb --z_scale 10 --terminal_std 0.001 --learning_rate 0.01 --force &
CUDA_VISIBLE_DEVICES=2 python src/train.py --wandb --z_scale 10 --terminal_std 0.001 --learning_rate 0.001 --force &
CUDA_VISIBLE_DEVICES=3 python src/train.py --wandb --z_scale 10 --terminal_std 0.001 --learning_rate 0.0001 --force &
CUDA_VISIBLE_DEVICES=4 python src/train.py --wandb --z_scale 10 --terminal_std 0.01 --learning_rate 0.1 --force &
CUDA_VISIBLE_DEVICES=5 python src/train.py --wandb --z_scale 10 --terminal_std 0.01 --learning_rate 0.01 --force &
CUDA_VISIBLE_DEVICES=6 python src/train.py --wandb --z_scale 10 --terminal_std 0.01 --learning_rate 0.001 --force &
CUDA_VISIBLE_DEVICES=7 python src/train.py --wandb --z_scale 10 --terminal_std 0.01 --learning_rate 0.0001 --force
