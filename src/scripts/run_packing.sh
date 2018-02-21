#!/bin/bash
# Stores only the masks after training.
# Usage:
# ./scripts/run_packing.sh 0 vgg16 binary

GPU_ID=$1
NAME=$2
TYPE=$3

CUDA_VISIBLE_DEVICES=$GPU_ID python pack.py --mode pack \
  --packlist best_models/$NAME'_'$TYPE'.txt' \
  --maskloc ../checkpoints/packed/$NAME'_'$TYPE'.pt'
