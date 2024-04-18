CUDA_VISIBLE_DEVICES=0 python src/train.py --project alanine

CUDA_VISIBLE_DEVICES=0 python src/train.py --molecule chignolin --project chignolin --start_std 0.15 --trains_per_sample 200 --terminal_std 0.5

CUDA_VISIBLE_DEVICES=0 python src/train.py --molecule poly --project poly --start_std 0.15 --trains_per_sample 1000 --terminal_std 0.1