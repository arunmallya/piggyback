## Piggyback: https://arxiv.org/abs/1801.06519

Pretrained masks and backbones are available here: 
Datasets in PyTorch format are available here: https://uofi.box.com/s/ixncr3d85guosajywhf7yridszzg5zsq 
The PyTorch-friendly Places365 dataset can be downloaded from http://places2.csail.mit.edu/download.html 
Place masks in `checkpoints/` and unzipped datasets in `data/`

|               |    VGG-16    |   VGG-16 BN  |   ResNet-50  | DenseNet-121 |
|:-------------:|:------------:|:------------:|:------------:|:------------:|
| CUBS          |              |              |              |              |
| Stanford Cars |              |              |              |              |
| Flowers       |              |              |              |              |
| WikiArt       |              |              |              |              |
| Sketch        |              |              |              |              |

Note that the numbers in the [paper](https://arxiv.org/abs/1801.06519) are averaged over multiple runs for each ordering
of datasets. 
These numbers were obtained by evaluating the models on a Titan X (Pascal). 
Note that numbers on other GPUs might be slightly different (~0.1%) owing to cudnn algorithm selection. 
https://discuss.pytorch.org/t/slightly-different-results-on-k-40-v-s-titan-x/10064

## Requirements:
torch==0.2.0.post3
torchvision==0.1.9


Run all code from the `src/` directory, e.g. `./scripts/run_piggyback_training.sh`

## Training:
Check out `src/scripts/run_piggyback_training.sh`.
This script uses the default hyperparams and trains a model as described in the paper. The best performing model on the val set
is saved to disk. This saved model includes the real-valued mask weights.

## Saving trained masks only.
Check out `src/scripts/run_packing.sh`.
This extracts the binary/ternary masks from the above trained models, and saves them separately.

## Eval:
Use the saved masks, apply them to a backbone network and run eval.
By default, our backbone models are those provided with pytorch.
Note that to replicate our results, you have to use the package versions specified above. 
Newer package versions might have different weights for the backbones, and the provided masks won't work.
```bash
cd src  # Run everything from src/

CUDA_VISIBLE_DEVICES=0 python pack.py --mode eval --dataset flowers \
  --arch vgg16 \
  --maskloc ../checkpoints/vgg16_binary.pt
```