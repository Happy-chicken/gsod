module purge
module load GCC/13.3.0 CUDA/12.4.0
export CC=gcc CXX=g++
export CUDA_HOME=$(dirname $(dirname $(which nvcc)))
source $HOME/miniforge3/bin/activate gsod
