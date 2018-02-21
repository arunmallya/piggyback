## Piggyback: https://arxiv.org/abs/1801.06519

Pretrained masks and backbones are available here: https://uofi.box.com/s/c5kixsvtrghu9yj51yb1oe853ltdfz4q

Datasets in PyTorch format are available here: https://uofi.box.com/s/ixncr3d85guosajywhf7yridszzg5zsq  
All rights belong to the respective publishers. The datasets are provided only to aid reproducibility.

The PyTorch-friendly Places365 dataset can be downloaded from http://places2.csail.mit.edu/download.html 

Place masks in `checkpoints/` and unzipped datasets in `data/`

|               |    VGG-16    |   ResNet-50  | DenseNet-121 |
|:-------------:|:------------:|:------------:|:------------:|
| CUBS          |         20.75|         18.23|         19.24|
| Stanford Cars |         11.78|         10.19|         10.62|
| Flowers       |          6.93|          4.77|          4.91|
| WikiArt       |         29.80|         28.57|         29.33|
| Sketch        |         22.30|         19.75|         20.05|

Note that the numbers in the [paper](https://arxiv.org/abs/1801.06519) are averaged over multiple runs for each ordering
of datasets. 
These numbers were obtained by evaluating the models on a Titan X (Pascal). 
Note that numbers on other GPUs might be slightly different (~0.1%) owing to cudnn algorithm selection. 
https://discuss.pytorch.org/t/slightly-different-results-on-k-40-v-s-titan-x/10064

## Requirements:
```
Python 2.7 or 3.xx
torch==0.2.0.post3
torchvision==0.1.9
torchnet (pip install git+https://github.com/pytorch/tnt.git@master)
tqdm (pip install tqdm)
```


Run all code from the `src/` directory, e.g. `./scripts/run_piggyback_training.sh`

## Training:
Check out `src/scripts/run_piggyback_training.sh`.

This script uses the default hyperparams and trains a model as described in the paper. The best performing model on the val set is saved to disk. This saved model includes the real-valued mask weights.

By default, we use the models provided by `torchvision` as our backbone networks. If you intend to evaluate with the masks provided by us, please use the correct version of `torch` and `torchvision`. In case you want to use a different version, but still want to use our masks, then download the `pytorch_backbone` networks provided in the box link above. Make appropriate changes to your pytorch code to load those backbone models.

## Saving trained masks only.
Check out `src/scripts/run_packing.sh`.

This extracts the binary/ternary masks from the above trained models, and saves them separately.

## Eval:
Use the saved masks, apply them to a backbone network and run eval.

By default, our backbone models are those provided with `torchvision`.  
Note that to replicate our results, you have to use the package versions specified above.  
Newer package versions might have different weights for the backbones, and the provided masks won't work.
```bash
cd src  # Run everything from src/

CUDA_VISIBLE_DEVICES=0 python pack.py --mode eval --dataset flowers \
  --arch vgg16 \
  --maskloc ../checkpoints/vgg16_binary.pt
```
