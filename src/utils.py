"""Contains a bunch of utility functions."""

import numpy as np


def step_lr(epoch, base_lr, lr_decay_every, lr_decay_factor, optimizer):
    """Handles step decay of learning rate."""
    factor = np.power(lr_decay_factor, np.floor((epoch - 1) / lr_decay_every))
    new_lr = base_lr * factor
    for param_group in optimizer.param_groups:
        param_group['lr'] = new_lr
    print('Set lr to ', new_lr)
    return optimizer


def set_dataset_paths(args):
    """Set default train and test path if not provided as input."""
    if not args.train_path:
        args.train_path = '../data/%s/train' % (args.dataset)
    if not args.test_path:
        if args.dataset == 'imagenet' or args.dataset == 'places':
            args.test_path = '../data/%s/val' % (args.dataset)
        else:
            args.test_path = '../data/%s/test' % (args.dataset)
