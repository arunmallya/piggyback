#!/bin/bash
# Runs the piggyback method using default settings.
# Usage:
# ./scripts/run_piggyback_training.sh vgg16 3 1 binarizer

# This is hard-coded to prevent silly mistakes.
declare -A NUM_OUTPUTS
NUM_OUTPUTS["imagenet"]="1000"
NUM_OUTPUTS["places"]="365"
NUM_OUTPUTS["stanford_cars_cropped"]="196"
NUM_OUTPUTS["cubs_cropped"]="200"
NUM_OUTPUTS["flowers"]="102"
NUM_OUTPUTS["wikiart"]="195"
NUM_OUTPUTS["sketches"]="250"

ARCH=$1
GPU_ID=$2
NUM_RUNS=$3
THRESHOLD_FN=$4
MASK_SCALE=1e-2
MASK_SCALE_GRADS=none
LR_MASK=1e-4
LR_CLASS=1e-4
MASK_DECAY=15
CLASS_DECAY=15
NUM_EPOCHS=30

for RUN_ID in `seq 1 $NUM_RUNS`;
do
  for DATASET in stanford_cars_cropped cubs_cropped flowers wikiart sketches; do
    mkdir ../checkpoints/$DATASET
    mkdir ../logs/$DATASET

    TAG=$ARCH'_'$THRESHOLD_FN'_maskscale'$MASK_SCALE'-'$MASK_SCALE_GRADS'_lr'$LR_MASK'-'$LR_CLASS'_decay'$MASK_DECAY'-'$CLASS_DECAY'_'$RUN_ID

    CKPT_DIR='../checkpoints/'$DATASET'/final/'
    mkdir $CKPT_DIR
    LOG_DIR='../logs/'$DATASET'/final/'
    mkdir $LOG_DIR

    CUDA_VISIBLE_DEVICES=$GPU_ID python main.py --mode finetune \
      --arch $ARCH --threshold_fn $THRESHOLD_FN \
      --mask_scale $MASK_SCALE --mask_scale_gradients $MASK_SCALE_GRADS \
      --dataset $DATASET --num_outputs ${NUM_OUTPUTS[$DATASET]} '--mask_adam' \
      --lr_mask $LR_MASK --lr_mask_decay_every $MASK_DECAY \
      --lr_classifier $LR_CLASS --lr_classifier_decay_every $CLASS_DECAY \
      --lr_decay_factor 0.1 --finetune_epochs $NUM_EPOCHS \
      --save_prefix $CKPT_DIR$TAG'.pt' | tee $LOG_DIR$TAG'.txt'
  done
done
