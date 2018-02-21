"""Main entry point for doing all stuff."""
from __future__ import division, print_function

import argparse
import json
import warnings

import torch
import torch.nn as nn
import torch.optim as optim
import torchnet as tnt
from torch.autograd import Variable
from tqdm import tqdm

import dataset
import networks as net
import utils as utils


# To prevent PIL warnings.
warnings.filterwarnings("ignore")

FLAGS = argparse.ArgumentParser()
FLAGS.add_argument('--arch',
                   choices=['vgg16', 'vgg16bn', 'resnet50',
                            'densenet121', 'resnet50_diff'],
                   help='Architectures')
FLAGS.add_argument('--source', type=str, default='',
                   help='Location of the init file for resnet50_diff')
FLAGS.add_argument('--finetune_layers',
                   choices=['all', 'fc', 'classifier'], default='all',
                   help='Which layers to finetune, fc only works with vgg')
FLAGS.add_argument('--mode',
                   choices=['finetune', 'eval', 'check'],
                   help='Run mode')
FLAGS.add_argument('--num_outputs', type=int, default=-1,
                   help='Num outputs for dataset')
# Optimization options.
FLAGS.add_argument('--lr', type=float,
                   help='Learning rate for parameters, used for baselines')
FLAGS.add_argument('--lr_decay_every', type=int,
                   help='Step decay every this many epochs')
FLAGS.add_argument('--lr_mask', type=float,
                   help='Learning rate for mask')
FLAGS.add_argument('--lr_mask_decay_every', type=int,
                   help='Step decay every this many epochs')
FLAGS.add_argument('--mask_adam', action='store_true', default=False,
                   help='Use adam instead of sgdm for masks')
FLAGS.add_argument('--lr_classifier', type=float,
                   help='Learning rate for classifier')
FLAGS.add_argument('--lr_classifier_decay_every', type=int,
                   help='Step decay every this many epochs')

FLAGS.add_argument('--lr_decay_factor', type=float,
                   help='Multiply lr by this much every step of decay')
FLAGS.add_argument('--finetune_epochs', type=int,
                   help='Number of initial finetuning epochs')
FLAGS.add_argument('--batch_size', type=int, default=32,
                   help='Batch size')
FLAGS.add_argument('--weight_decay', type=float, default=0.0,
                   help='Weight decay')
FLAGS.add_argument('--train_bn', action='store_true', default=False,
                   help='train batch norm or not')
# Masking options.
FLAGS.add_argument('--mask_init', default='1s',
                   choices=['1s', 'uniform', 'weight_based_1s'],
                   help='Type of mask init')
FLAGS.add_argument('--mask_scale', type=float, default=1e-2,
                   help='Mask initialization scaling')
FLAGS.add_argument('--mask_scale_gradients', type=str, default='none',
                   choices=['none', 'average', 'individual'],
                   help='Scale mask gradients by weights')
FLAGS.add_argument('--threshold_fn',
                   choices=['binarizer', 'ternarizer'],
                   help='Type of thresholding function')
# Paths.
FLAGS.add_argument('--dataset', type=str, default='',
                   help='Name of dataset')
FLAGS.add_argument('--train_path', type=str, default='',
                   help='Location of train data')
FLAGS.add_argument('--test_path', type=str, default='',
                   help='Location of test data')
FLAGS.add_argument('--save_prefix', type=str, default='../checkpoints/',
                   help='Location to save model')
FLAGS.add_argument('--loadname', type=str, default='',
                   help='Location to save model')
# Other.
FLAGS.add_argument('--cuda', action='store_true', default=True,
                   help='use CUDA')
FLAGS.add_argument('--no_mask', action='store_true', default=False,
                   help='Used for running baselines, does not use any masking')


class Manager(object):
    """Handles training and pruning."""

    def __init__(self, args, model):
        self.args = args
        self.cuda = args.cuda
        self.model = model

        # Set up data loader, criterion, and pruner.
        if 'cropped' in args.train_path:
            train_loader = dataset.train_loader_cropped
            test_loader = dataset.test_loader_cropped
        else:
            train_loader = dataset.train_loader
            test_loader = dataset.test_loader
        self.train_data_loader = train_loader(
            args.train_path, args.batch_size, pin_memory=args.cuda)
        self.test_data_loader = test_loader(
            args.test_path, args.batch_size, pin_memory=args.cuda)
        self.criterion = nn.CrossEntropyLoss()

    def eval(self):
        """Performs evaluation."""
        self.model.eval()
        error_meter = None

        print('Performing eval...')
        for batch, label in tqdm(self.test_data_loader, desc='Eval'):
            if self.cuda:
                batch = batch.cuda()
            batch = Variable(batch, volatile=True)

            output = self.model(batch)

            # Init error meter.
            if error_meter is None:
                topk = [1]
                if output.size(1) > 5:
                    topk.append(5)
                error_meter = tnt.meter.ClassErrorMeter(topk=topk)
            error_meter.add(output.data, label)

        errors = error_meter.value()
        print('Error: ' + ', '.join('@%s=%.2f' %
                                    t for t in zip(topk, errors)))

        if 'train_bn' in self.args:
            if self.args.train_bn:
                self.model.train()
            else:
                self.model.train_nobn()
        else:
            print('args does not have train_bn flag, probably in eval-only mode.')
        return errors

    def do_batch(self, optimizer, batch, label):
        """Runs model for one batch."""
        if self.cuda:
            batch = batch.cuda()
            label = label.cuda()
        batch = Variable(batch)
        label = Variable(label)

        # Set grads to 0.
        self.model.zero_grad()

        # Do forward-backward.
        output = self.model(batch)
        self.criterion(output, label).backward()

        # Scale gradients by average weight magnitude.
        if self.args.mask_scale_gradients != 'none':
            for module in self.model.shared.modules():
                if 'ElementWise' in str(type(module)):
                    abs_weights = module.weight.data.abs()
                    if self.args.mask_scale_gradients == 'average':
                        module.mask_real.grad.data.div_(abs_weights.mean())
                    elif self.args.mask_scale_gradients == 'individual':
                        module.mask_real.grad.data.div_(abs_weights)

        # Set batchnorm grads to 0, if required.
        if not self.args.train_bn:
            for module in self.model.shared.modules():
                if 'BatchNorm' in str(type(module)):
                    if module.weight.grad is not None:
                        module.weight.grad.data.fill_(0)
                    if module.bias.grad is not None:
                        module.bias.grad.data.fill_(0)

        # Update params.
        optimizer.step()

    def do_epoch(self, epoch_idx, optimizer):
        """Trains model for one epoch."""
        for batch, label in tqdm(self.train_data_loader, desc='Epoch: %d ' % (epoch_idx)):
            self.do_batch(optimizer, batch, label)

        if self.args.threshold_fn == 'binarizer':
            print('Num 0ed out parameters:')
            for idx, module in enumerate(self.model.shared.modules()):
                if 'ElementWise' in str(type(module)):
                    num_zero = module.mask_real.data.lt(5e-3).sum()
                    total = module.mask_real.data.numel()
                    print(idx, num_zero, total)
        elif self.args.threshold_fn == 'ternarizer':
            print('Num -1, 0ed out parameters:')
            for idx, module in enumerate(self.model.shared.modules()):
                if 'ElementWise' in str(type(module)):
                    num_neg = module.mask_real.data.lt(0).sum()
                    num_zero = module.mask_real.data.lt(5e-3).sum() - num_neg
                    total = module.mask_real.data.numel()
                    print(idx, num_neg, num_zero, total)
        print('-' * 20)

    def save_model(self, epoch, best_accuracy, errors, savename):
        """Saves model to file."""
        # Prepare the ckpt.
        ckpt = {
            'args': self.args,
            'epoch': epoch,
            'accuracy': best_accuracy,
            'errors': errors,
            'model': self.model,
        }

        # Save to file.
        torch.save(ckpt, savename)

    def train(self, epochs, optimizer, save=True, savename='', best_accuracy=0):
        """Performs training."""
        best_accuracy = best_accuracy
        error_history = []

        if self.args.cuda:
            self.model = self.model.cuda()

        self.eval()

        for idx in range(epochs):
            epoch_idx = idx + 1
            print('Epoch: %d' % (epoch_idx))

            optimizer.update_lr(epoch_idx)
            if self.args.train_bn:
                self.model.train()
            else:
                self.model.train_nobn()
            self.do_epoch(epoch_idx, optimizer)
            errors = self.eval()
            error_history.append(errors)
            accuracy = 100 - errors[0]  # Top-1 accuracy.

            # Save performance history and stats.
            with open(savename + '.json', 'w') as fout:
                json.dump({
                    'error_history': error_history,
                    'args': vars(self.args),
                }, fout)

            # Save best model, if required.
            if save and accuracy > best_accuracy:
                print('Best model so far, Accuracy: %0.2f%% -> %0.2f%%' %
                      (best_accuracy, accuracy))
                best_accuracy = accuracy
                self.save_model(epoch_idx, best_accuracy, errors, savename)

        # Make sure masking didn't change any weights.
        if not self.args.no_mask:
            self.check()
        print('Finished finetuning...')
        print('Best error/accuracy: %0.2f%%, %0.2f%%' %
              (100 - best_accuracy, best_accuracy))
        print('-' * 16)

    def check(self):
        """Makes sure that the trained model weights match those of the pretrained model."""
        print('Making sure filter weights have not changed.')
        if self.args.arch == 'vgg16':
            pretrained = net.ModifiedVGG16(original=True)
        elif self.args.arch == 'vgg16bn':
            pretrained = net.ModifiedVGG16BN(original=True)
        elif self.args.arch == 'resnet50':
            pretrained = net.ModifiedResNet(original=True)
        elif self.args.arch == 'densenet121':
            pretrained = net.ModifiedDenseNet(original=True)
        elif self.args.arch == 'resnet50_diff':
            pretrained = net.ResNetDiffInit(self.args.source, original=True)
        else:
            raise ValueError('Architecture %s not supported.' %
                             (self.args.arch))

        for module, module_pretrained in zip(self.model.shared.modules(), pretrained.shared.modules()):
            if 'ElementWise' in str(type(module)) or 'BatchNorm' in str(type(module)):
                weight = module.weight.data.cpu()
                weight_pretrained = module_pretrained.weight.data.cpu()
                # Using small threshold of 1e-8 for any floating point inconsistencies.
                # Note that threshold per element is even smaller as the 1e-8 threshold
                # is for sum of absolute differences.
                assert (weight - weight_pretrained).abs().sum() < 1e-8, \
                    'module %s failed check' % (module)
                if module.bias is not None:
                    bias = module.bias.data.cpu()
                    bias_pretrained = module_pretrained.bias.data.cpu()
                    assert (bias - bias_pretrained).abs().sum() < 1e-8
                if 'BatchNorm' in str(type(module)):
                    rm = module.running_mean.cpu()
                    rm_pretrained = module_pretrained.running_mean.cpu()
                    assert (rm - rm_pretrained).abs().sum() < 1e-8
                    rv = module.running_var.cpu()
                    rv_pretrained = module_pretrained.running_var.cpu()
                    assert (rv - rv_pretrained).abs().sum() < 1e-8
        print('Passed checks...')


class Optimizers(object):
    """Handles a list of optimizers."""

    def __init__(self, args):
        self.optimizers = []
        self.lrs = []
        self.decay_every = []
        self.args = args

    def add(self, optimizer, lr, decay_every):
        """Adds optimizer to list."""
        self.optimizers.append(optimizer)
        self.lrs.append(lr)
        self.decay_every.append(decay_every)

    def step(self):
        """Makes all optimizers update their params."""
        for optimizer in self.optimizers:
            optimizer.step()

    def update_lr(self, epoch_idx):
        """Update learning rate of every optimizer."""
        for optimizer, init_lr, decay_every in zip(self.optimizers, self.lrs, self.decay_every):
            optimizer = utils.step_lr(
                epoch_idx, init_lr, decay_every,
                self.args.lr_decay_factor, optimizer)


def main():
    """Do stuff."""
    args = FLAGS.parse_args()

    # Set default train and test path if not provided as input.
    utils.set_dataset_paths(args)

    # Load the required model.
    if args.arch == 'vgg16':
        model = net.ModifiedVGG16(mask_init=args.mask_init,
                                  mask_scale=args.mask_scale,
                                  threshold_fn=args.threshold_fn,
                                  original=args.no_mask)
    elif args.arch == 'vgg16bn':
        model = net.ModifiedVGG16BN(mask_init=args.mask_init,
                                    mask_scale=args.mask_scale,
                                    threshold_fn=args.threshold_fn,
                                    original=args.no_mask)
    elif args.arch == 'resnet50':
        model = net.ModifiedResNet(mask_init=args.mask_init,
                                   mask_scale=args.mask_scale,
                                   threshold_fn=args.threshold_fn,
                                   original=args.no_mask)
    elif args.arch == 'densenet121':
        model = net.ModifiedDenseNet(mask_init=args.mask_init,
                                     mask_scale=args.mask_scale,
                                     threshold_fn=args.threshold_fn,
                                     original=args.no_mask)
    elif args.arch == 'resnet50_diff':
        assert args.source
        model = net.ResNetDiffInit(args.source,
                                   mask_init=args.mask_init,
                                   mask_scale=args.mask_scale,
                                   threshold_fn=args.threshold_fn,
                                   original=args.no_mask)
    else:
        raise ValueError('Architecture %s not supported.' % (args.arch))

    # Add and set the model dataset.
    model.add_dataset(args.dataset, args.num_outputs)
    model.set_dataset(args.dataset)
    if args.cuda:
        model = model.cuda()

    # Initialize with weight based method, if necessary.
    if not args.no_mask and args.mask_init == 'weight_based_1s':
        print('Are you sure you want to try this?')
        assert args.mask_scale_gradients == 'none'
        assert not args.mask_scale
        for idx, module in enumerate(model.shared.modules()):
            if 'ElementWise' in str(type(module)):
                weight_scale = module.weight.data.abs().mean()
                module.mask_real.data.fill_(weight_scale)

    # Create the manager object.
    manager = Manager(args, model)

    # Perform necessary mode operations.
    if args.mode == 'finetune':
        if args.no_mask:
            # No masking will be done, used to run baselines of
            # Classifier-Only and Individual Networks.
            # Checks.
            assert args.lr and args.lr_decay_every
            assert not args.lr_mask and not args.lr_mask_decay_every
            assert not args.lr_classifier and not args.lr_classifier_decay_every
            print('No masking, running baselines.')

            # Get optimizer with correct params.
            if args.finetune_layers == 'all':
                params_to_optimize = model.parameters()
            elif args.finetune_layers == 'classifier':
                for param in model.shared.parameters():
                    param.requires_grad = False
                params_to_optimize = model.classifier.parameters()

            # optimizer = optim.Adam(params_to_optimize, lr=args.lr)
            optimizer = optim.SGD(params_to_optimize, lr=args.lr,
                                  momentum=0.9, weight_decay=args.weight_decay)
            optimizers = Optimizers(args)
            optimizers.add(optimizer, args.lr, args.lr_decay_every)
            manager.train(args.finetune_epochs, optimizers,
                          save=True, savename=args.save_prefix)
        else:
            # Masking will be done.
            # Checks.
            assert not args.lr and not args.lr_decay_every
            assert args.lr_mask and args.lr_mask_decay_every
            assert args.lr_classifier and args.lr_classifier_decay_every
            print('Performing masking.')

            optimizer_masks = optim.Adam(
                model.shared.parameters(), lr=args.lr_mask)
            optimizer_classifier = optim.Adam(
                model.classifier.parameters(), lr=args.lr_classifier)

            optimizers = Optimizers(args)
            optimizers.add(optimizer_masks, args.lr_mask,
                           args.lr_mask_decay_every)
            optimizers.add(optimizer_classifier, args.lr_classifier,
                           args.lr_classifier_decay_every)
            manager.train(args.finetune_epochs, optimizers,
                          save=True, savename=args.save_prefix)
    elif args.mode == 'eval':
        # Just run the model on the eval set.
        manager.eval()
    elif args.mode == 'check':
        manager.check()


if __name__ == '__main__':
    main()
