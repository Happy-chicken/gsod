# CLAIX environment for gsod; source it (login node or job), never run it:
#     source scripts/hpc_env.sh
# GCC, not the default Intel compiler (PyTorch is built with g++, and gsplat
# compiles its extension with $CXX); CUDA matching torch.version.cuda. Keep
# the versions the gsplat extension was compiled with.
module purge
module load GCC/13.3.0 CUDA/12.4.0
export CC=gcc CXX=g++
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
source $HOME/miniforge3/bin/activate gsod
