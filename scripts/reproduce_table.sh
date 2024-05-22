# CUDA_VISIBLE_DEVICES=0 python src/eval.py --model_path model/alanine/pot_fix.pt --bias_scale 20000

# CUDA_VISIBLE_DEVICES=1 python src/eval.py --model_path model/alanine/force_fix.pt --force --bias_scale 1000


CUDA_VISIBLE_DEVICES=0 python src/eval.py --model_path model/histidine/histidine_table.pt --sigma 0.1 --num_steps 1000 --molecule histidine --num_samples 4