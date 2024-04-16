CUDA_VISIBLE_DEVICES=0 python src/train.py --project alanine --wandb --force --terminal_std 1 --learning_rate 0.1 &
CUDA_VISIBLE_DEVICES=1 python src/train.py --project alanine --wandb --force --terminal_std 0.1 --learning_rate 0.1 &
CUDA_VISIBLE_DEVICES=2 python src/train.py --project alanine --wandb --force --terminal_std 0.01 --learning_rate 0.1 &
CUDA_VISIBLE_DEVICES=3 python src/train.py --project alanine --wandb --force --terminal_std 10 --learning_rate 0.1 &
CUDA_VISIBLE_DEVICES=4 python src/train.py --project alanine --wandb --force --terminal_std 1 --learning_rate 0.01 &
CUDA_VISIBLE_DEVICES=5 python src/train.py --project alanine --wandb --force --terminal_std 0.1 --learning_rate 0.01 &
CUDA_VISIBLE_DEVICES=6 python src/train.py --project alanine --wandb --force --terminal_std 0.01 --learning_rate 0.01 &
CUDA_VISIBLE_DEVICES=7 python src/train.py --project alanine --wandb --force --terminal_std 10 --learning_rate 0.01
