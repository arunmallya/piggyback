"""Packs binary masks only, showing that we're not changing pre-trained weights.

  Usage for packing masks only:
  CUDA_VISIBLE_DEVICES=$GPU_ID python pack.py --mode pack \
    --packlist best_models/densenet121_binary.txt \
    --maskloc ../checkpoints/packed/densenet121_binary

  Usage for eval:
  CUDA_VISIBLE_DEVICES=0 python pack.py --mode eval --arch densenet121 \
    --maskloc ../checkpoints/packed/densenet121_binary.pt \
    --dataset cubs_cropped
"""

from __future__ import division, print_function

import argparse

import torch

import networks as net
from main import Manager
import utils as utils


FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--mode',
                   choices=['pack', 'eval'],
                   help='Run mode')
# Packing arguments.
FLAGS.add_argument('--packlist', type=str,
                   help='File containing dataset:model per line')
# Eval arguments.
FLAGS.add_argument('--arch',
                   choices=['vgg16', 'vgg16bn', 'resnet50',
                            'densenet121', 'resnet50_diff'],
                   help='Type of architecture')
FLAGS.add_argument('--source', type=str, default='',
                   help='Location to load model from for resnet50_diff')
FLAGS.add_argument('--maskloc', type=str, default='',
                   help='Location to save/load masks from')
FLAGS.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')
FLAGS.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
FLAGS.add_argument('--test_path', type=str, default='',
                   help='Location of test data')
FLAGS.add_argument('--batch_size', type=int, default=32,
                   help='Batch size')
# Other.
FLAGS.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')


def main():
    """Do stuff."""
    args = FLAGS.parse_args()

    if args.mode == 'pack':
        assert args.packlist and args.maskloc
        dataset2masks = {}
        dataset2classifiers = {}
        net_type = None

        # Location to output stats.
        fout = open(args.maskloc[:-2] + 'txt', 'w')

        # Load models one by one and store their masks.
        fin = open(args.packlist, 'r')
        counter = 1
        for idx, line in enumerate(fin):
            if not line or not line.strip() or line[0] == '#':
                continue
            dataset, loadname = line.split(':')
            loadname = loadname.strip()

            # Can't have same dataset twice.
            if dataset in dataset2masks:
                ValueError('Repeated datasets as input...')
            print('Loading model #%d for dataset "%s"' % (counter, dataset))
            counter += 1
            ckpt = torch.load(loadname)
            model = ckpt['model']
            # Ensure all inputs are for same model type.
            if net_type is None:
                net_type = str(type(model))
            else:
                assert net_type == str(type(model)), '%s != %s' % (
                    net_type, str(type(model)))

            # Gather masks and store in dictionary.
            fout.write('Dataset: %s\n' % (dataset))
            total_params, neg_params, zerod_params = [], [], []
            masks = {}
            for module_idx, module in enumerate(model.shared.modules()):
                if 'ElementWise' in str(type(module)):
                    mask = module.threshold_fn(module.mask_real)
                    mask = mask.data.cpu()

                    # Make sure mask values are in {0, 1} or {-1, 0, 1}.
                    num_zero = mask.eq(0).sum()
                    num_one = mask.eq(1).sum()
                    num_mone = mask.eq(-1).sum()
                    total = mask.numel()
                    threshold_type = module.threshold_fn.__class__.__name__
                    if threshold_type == 'Binarizer':
                        assert num_mone == 0
                        assert num_zero + num_one == total
                    elif threshold_type == 'Ternarizer':
                        assert num_mone + num_zero + num_one == total
                    masks[module_idx] = mask.type(torch.ByteTensor)

                    # Count total and zerod out params.
                    total_params.append(total)
                    zerod_params.append(num_zero)
                    neg_params.append(num_mone)
                    fout.write('%d\t%.2f%%\t%.2f%%\n' % (
                        module_idx,
                        neg_params[-1] / total_params[-1] * 100,
                        zerod_params[-1] / total_params[-1] * 100))
            print('Check Passed: Masks only have binary/ternary values.')
            dataset2masks[dataset] = masks
            dataset2classifiers[dataset] = model.classifier

            fout.write('Total -1: %d/%d = %.2f%%\n' % (
                sum(neg_params), sum(total_params), sum(neg_params) / sum(total_params) * 100))
            fout.write('Total 0: %d/%d = %.2f%%\n' % (
                sum(zerod_params), sum(total_params), sum(zerod_params) / sum(total_params) * 100))
            fout.write('-' * 20 + '\n')

        # Clean up and save masks to file.
        fin.close()
        fout.close()
        torch.save({
            'dataset2masks': dataset2masks,
            'dataset2classifiers': dataset2classifiers,
        }, args.maskloc)

    elif args.mode == 'eval':
        assert args.arch and args.maskloc and args.dataset

        # Set default train and test path if not provided as input.
        utils.set_dataset_paths(args)

        # Load masks and classifier for this dataset.
        info = torch.load(args.maskloc)
        if args.dataset not in info['dataset2masks']:
            ValueError('%s not found in masks.' % (args.dataset))
        masks = info['dataset2masks'][args.dataset]
        classifier = info['dataset2classifiers'][args.dataset]

        # Create the vanilla model and apply masking.
        model = None
        if args.arch == 'vgg16':
            model = net.ModifiedVGG16(original=True)
        elif args.arch == 'vgg16bn':
            model = net.ModifiedVGG16BN(original=True)
        elif args.arch == 'resnet50':
            model = net.ModifiedResNet(original=True)
        elif args.arch == 'densenet121':
            model = net.ModifiedDenseNet(original=True)
        elif args.arch == 'resnet50_diff':
            assert args.source
            model = net.ResNetDiffInit(args.source, original=True)
        model.eval()

        print('Applying masks.')
        for module_idx, module in enumerate(model.shared.modules()):
            if module_idx in masks:
                mask = masks[module_idx]
                module.weight.data[mask.eq(0)] = 0
                module.weight.data[mask.eq(-1)] *= -1
        print('Applied masks.')

        # Override model.classifier with saved one.
        model.add_dataset(args.dataset, classifier.weight.size(0))
        model.set_dataset(args.dataset)
        model.classifier = classifier
        if args.cuda:
            model = model.cuda()

        # Create the manager and run eval.
        manager = Manager(args, model)
        manager.eval()


if __name__ == '__main__':
    main()
