#!/usr/bin/zsh
# Slurm job for CLAIX (RWTH). Fill in the account; check the partition,
# module names and limits against the RWTH HPC docs. Submit from the repo:
#     mkdir -p logs && sbatch scripts/slurm_train.sh
# Extra train.py arguments go after the script: sbatch scripts/slurm_train.sh --epochs 80
#SBATCH --job-name=gsod
#SBATCH --account=CHANGE_ME
#SBATCH --partition=c23g
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out

module purge
module load CUDA  # `module spider CUDA` lists the versions; gsplat needs one matching PyTorch's
source $HOME/miniforge3/bin/activate gsod

cd $SLURM_SUBMIT_DIR
export WANDB_MODE=offline  # compute nodes have no internet; `wandb sync` from a login node later
export WANDB_DIR=$HPCWORK/wandb
export TMPDIR=$TMP

python -u train.py --dataset room_diverse \
    --train-scenes 800 --val-scenes 50 --monitor-scenes 20 --epochs 80 --eval-every 4 \
    --out $HPCWORK/runs/room_diverse_$SLURM_JOB_ID "$@"
