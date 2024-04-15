CUDA_VISIBLE_DEVICES=0 python src/train.py --seed 0 --project alanine_test --wandb --force --bias_scale 100 --bias --start_temp 600 &
CUDA_VISIBLE_DEVICES=1 python src/train.py --seed 0 --project alanine_test --wandb --force --bias_scale 100 --bias &
CUDA_VISIBLE_DEVICES=2 python src/train.py --seed 0 --project alanine_test --wandb --force --bias_scale 100 --start_temp 600 &
CUDA_VISIBLE_DEVICES=3 python src/train.py --seed 0 --project alanine_test --wandb --force --bias_scale 100 &
CUDA_VISIBLE_DEVICES=4 python src/train.py --seed 0 --project alanine_test --wandb --force --bias --start_temp 600 &
CUDA_VISIBLE_DEVICES=5 python src/train.py --seed 0 --project alanine_test --wandb --force --bias &
CUDA_VISIBLE_DEVICES=6 python src/train.py --seed 0 --project alanine_test --wandb --force --start_temp 600 &
CUDA_VISIBLE_DEVICES=7 python src/train.py --seed 0 --project alanine_test --wandb --force  

# CUDA_VISIBLE_DEVICES=0 python src/train.py --seed 0 --project test --start_states c5,c7eq,c7ax --end_states c5,c7eq,c7ax --goal_conditioned &
# CUDA_VISIBLE_DEVICES=1 python alanine.py --seed 1 --project alanine_off_cond --start_states c5,c7eq,c7ax --end_states c5,c7eq,c7ax --goal_conditioned &
# CUDA_VISIBLE_DEVICES=2 python alanine.py --seed 2 --project alanine_off_cond --start_states c5,c7eq,c7ax --end_states c5,c7eq,c7ax --goal_conditioned &
# CUDA_VISIBLE_DEVICES=3 python alanine.py --seed 3 --project alanine_off_cond --start_states c5,c7eq,c7ax --end_states c5,c7eq,c7ax --goal_conditioned &
# CUDA_VISIBLE_DEVICES=4 python alanine.py --seed 4 --project alanine_off_cond --start_states c5,c7eq,c7ax --end_states c5,c7eq,c7ax --goal_conditioned &
# CUDA_VISIBLE_DEVICES=5 python alanine.py --seed 5 --project alanine_off_cond --start_states c5,c7eq,c7ax --end_states c5,c7eq,c7ax --goal_conditioned &
# CUDA_VISIBLE_DEVICES=6 python alanine.py --seed 6 --project alanine_off_cond --start_states c5,c7eq,c7ax --end_states c5,c7eq,c7ax --goal_conditioned &
# CUDA_VISIBLE_DEVICES=7 python alanine.py --seed 7 --project alanine_off_cond --start_states c5,c7eq,c7ax --end_states c5,c7eq,c7ax --goal_conditioned
