CUDA_VISIBLE_DEVICES=0 python src/eval.py --model_path model/chignolin/pot_flex.pt --project chignolin --molecule chignolin --start_state unfolded --end_state folded --num_steps 5000 --sigma 0.2 --num_samples 2 --unbiased_steps 5000

# CUDA_VISIBLE_DEVICES=1 python src/eval.py --model_path model/chignolin/force_flex.pt --molecule chignolin --start_state unfolded --end_state folded --num_steps 5000 --sigma 0.2 --num_samples 32 --force --bias_scale 0.00001

# current_date=$(date +"%m%d-%H%M%S")
# for seed in {5..5}; do
#   CUDA_VISIBLE_DEVICES=$seed python src/eval.py \
#     --project chignolin_unbiased \
#     --molecule chignolin \
#     --date 0623-231104 \
#     --seed $seed \
#     --start_state unfolded \
#     --end_state folded \
#     --sigma 0.2 \
#     --num_steps 10000 \
#     --unbiased_steps 5000 \
#     --num_samples 2
# done

# wait