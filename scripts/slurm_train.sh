#!/usr/bin/zsh
# Slurm job for CLAIX (RWTH). Fill in the account; check the partition,
# module names and limits against the RWTH HPC docs. Submit from the repo:
#     mkdir -p logs && sbatch scripts/slurm_train.sh
# Default: the whole training set, 100 epochs (~7 h on an H100). Extra train.py
# arguments go after the script, e.g. a quick run:
#     sbatch scripts/slurm_train.sh --train-scenes 800 --epochs 80 --eval-every 4
#SBATCH --job-name=gsod
#SBATCH --account=CHANGE_ME
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=08:00:00
#SBATCH --output=logs/%x_%j.out

cd $SLURM_SUBMIT_DIR
source scripts/hpc_env.sh  # modules, compilers, conda env
export WANDB_MODE=offline  # compute nodes have no internet; `wandb sync` from a login node later
export WANDB_DIR=$HPCWORK/wandb
export TMPDIR=$TMP

python -u train.py --dataset room_diverse \
    --val-scenes 50 --monitor-scenes 20 --epochs 100 --eval-every 5 --save-every 10 \
    --out $HPCWORK/runs/room_diverse_$SLURM_JOB_ID "$@"
