#!/usr/bin/env bash
# Point gsplat's JIT build at a CUDA 12.4 toolkit.
#
# Set CUDA_HOME to override; otherwise probe the usual locations. Some conda
# toolkits keep the CUDA headers/libs under targets/<arch>/ instead of
# include/ and lib64/, so add those to the compiler search paths as well.

if [[ -z "${CUDA_HOME:-}" ]]; then
    for candidate in /usr/local/cuda-12.4 /usr/local/cuda "$HOME/cuda-12.4"; do
        if [[ -x "$candidate/bin/nvcc" ]]; then
            export CUDA_HOME="$candidate"
            break
        fi
    done
fi

if [[ -n "${CUDA_HOME:-}" ]]; then
    export PATH="$CUDA_HOME/bin:$PATH"
    for target in "$CUDA_HOME"/targets/*/; do
        [[ -d "$target/include" ]] && export CPATH="$target/include${CPATH:+:$CPATH}"
        [[ -d "$target/lib" ]] && export LIBRARY_PATH="$target/lib${LIBRARY_PATH:+:$LIBRARY_PATH}"
    done
fi
