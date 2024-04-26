# !/bin/sh

current_date=$(date +"%m%d-%H%M%S")
echo $current_date

# project=$1
project=poly

target_dir=../cluster/poly/$project/train
if [ ! -d $target_dir ]; then
  mkdir -p $target_dir
fi


for seed in {0..7}
do
  for terminal_std in 0.2 0.4
  do
    for lr in 0.008 0.02 0.04
    do
      echo ">>" Training poly for seed $seed
      sbatch <<EOT
#!/bin/sh
#SBATCH -J hyun_poly
#SBATCH -o ../cluster/poly/$project/train/%j-seed$seed.out
#SBATCH -p A5000
#SBATCH -t 72:0:00
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
##SBATCH --nodelist=1
##SBATCH --cpus-per-task=1
##SBATCH --cpus-per-gpu=4

module purge
module load cuDNN/cuda/11.3 gnu12
module list

date
nvidia-smi

cd ../
WANDB__SERVICE_WAIT=300 python src/train.py \
  --molecule poly \
  --project $project \
  --date $current_date \
  --seed 0 \
  --wandb \
  --start_states pp2 \
  --end_states pp1 \
  --trains_per_rollout 1000 \
  --learning_rate $lr \
  --num_steps 5000 \
  --num_samples 2 \
  --start_std 0.1 \
  --end_std 0.05 \
  --terminal_std $terminal_std &
module purge
EOT
      sleep 0.2
    done
  done
done