#!/usr/bin/zsh
# CLAIX job: mkdir -p logs && sbatch scripts/slurm_train.sh [train.py args]
#SBATCH --job-name=gsod
#SBATCH --account=CHANGE_ME
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

cd $SLURM_SUBMIT_DIR
source scripts/hpc_env.sh
export WANDB_MODE=offline  # no internet on compute nodes
export WANDB_DIR=$HPCWORK/wandb
export TMPDIR=$TMP

python -u train.py --dataset room_diverse \
    --val-scenes 50 --monitor-scenes 20 --epochs 100 --eval-every 5 --save-every 10 \
    --out $HPCWORK/runs/room_diverse_$SLURM_JOB_ID "$@"
