#!/usr/bin/zsh
############################################################
### Slurm flags
############################################################

#SBATCH --partition=c23g            # request partition with GPU nodes
#SBATCH --nodes=1                   # request desired number of nodes
#SBATCH --ntasks-per-node=1         # request desired number of processes (or MPI tasks)

#SBATCH --cpus-per-task=24          # request desired number of CPU cores or threads per process (default: 1)
                                    # Note: available main memory is also scaling with
                                    #       number of cores if not specified otherwise
                                    # Note: On CLAIX-2023 each GPU can be used with 24 cores

#SBATCH --gres=gpu:1                # specify desired number of GPUs per node
#SBATCH --time=10:00:00             # max. run time of the job
#SBATCH --job-name=gsod		    # set the job name
#SBATCH --output=stdout_%j.txt      # redirects stdout and stderr to stdout.txt


cd $SLURM_SUBMIT_DIR
source scripts/hpc_env.sh  # modules, compilers, conda env
export WANDB_MODE=offline  # compute nodes have no internet; `wandb sync` from a login node later
export WANDB_DIR=$HPCWORK/wandb
export TMPDIR=$TMP

python -u train.py --dataset room_diverse \
    --val-scenes 50 --monitor-scenes 20 --epochs 100 --eval-every 5 --save-every 10 \
    --out $HPCWORK/runs/room_diverse_$SLURM_JOB_ID "$@"


