# for seed in {0..7}; do
#   CUDA_VISIBLE_DEVICES=$seed python src/eval.py \
#     --project 0707_chignolin_s_dist_heavy_dist_feat \
#     --molecule chignolin \
#     --date 0706-152536 \
#     --seed $seed \
#     --reward s_dist \
#     --heavy_atoms \
#     --start_state unfolded \
#     --end_state folded \
#     --feat_aug dist \
#     --wandb \
#     --sigma 300 \
#     --num_steps 10000 \
#     --num_samples 2 &
# done

# wait


  CUDA_VISIBLE_DEVICES=0 python src/eval.py \
    --project 0707_chignolin_s_dist_heavy_dist_feat \
    --molecule chignolin \
    --date 0706-152536 \
    --seed 0 \
    --reward s_dist \
    --heavy_atoms \
    --start_state unfolded \
    --end_state folded \
    --feat_aug dist \
    --sigma 300 \
    --num_steps 100 \
    --num_samples 2