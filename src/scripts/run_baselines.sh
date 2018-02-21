#!/bin/bash
# Runs the Classifier Only and Individual Model baselines.
# Usage:
# ./scripts/run_baselines.sh 3 3 vgg16
# ./scripts/run_baselines.sh 3 3 vgg16bn

# This is hard-coded to prevent silly mistakes.
declare -A NUM_OUTPUTS
NUM_OUTPUTS["imagenet"]="1000"
NUM_OUTPUTS["places"]="365"
NUM_OUTPUTS["stanford_cars_cropped"]="196"
NUM_OUTPUTS["cubs_cropped"]="200"
NUM_OUTPUTS["flowers"]="102"
NUM_OUTPUTS["wikiart"]="195"
NUM_OUTPUTS["sketches"]="250"

GPU_ID=$1
NUM_RUNS=$2
ARCH=$3
LR=1e-3

for RUN_ID in `seq 1 $NUM_RUNS`;
do
  for DATASET in stanford_cars_cropped cubs_cropped flowers wikiart sketches; do
    mkdir ../checkpoints/$DATASET
    mkdir ../logs/$DATASET

    # for FT_LAYERS in classifier all; do
    for FT_LAYERS in all; do
      if [ "$FT_LAYERS" == "classifier" ]; then
        LR_DECAY_EVERY=30
        TRAIN_BN=''
      else
        LR_DECAY_EVERY=15
        TRAIN_BN='--train_bn'
      fi

      LOG_DIR=../logs/$DATASET/'final_'$FT_LAYERS
      mkdir $LOG_DIR
      CKPT_DIR=../checkpoints/$DATASET/'final_'$FT_LAYERS
      mkdir $CKPT_DIR
      TAG=$ARCH'_SGD_lr'$LR'_lrdecay'$LR_DECAY_EVERY'_'$RUN_ID

      CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --mode finetune \
        --arch $ARCH \
        --dataset $DATASET --num_outputs ${NUM_OUTPUTS[$DATASET]} \
        --no_mask --finetune_layers $FT_LAYERS $TRAIN_BN \
        --lr $LR --lr_decay_every $LR_DECAY_EVERY \
        --lr_decay_factor 0.1 --finetune_epochs 30 \
        --save_prefix $CKPT_DIR'/'$TAG'.pt' | tee $LOG_DIR'/'$TAG'.txt'
    done
  done
done
