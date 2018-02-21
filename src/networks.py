"""Contains various network definitions."""
from __future__ import division, print_function

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torchvision import models

import modnets
import modnets.layers as nl


class View(nn.Module):
    """Changes view using a nn.Module."""

    def __init__(self, *shape):
        super(View, self).__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)


class ModifiedVGG16(nn.Module):
    """VGG16 with support for multiple classifiers."""

    def __init__(self, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False):
        super(ModifiedVGG16, self).__init__()
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original)

    def make_model(self, mask_init, mask_scale, threshold_fn, original):
        """Creates the model."""
        if original:
            vgg16 = models.vgg16(pretrained=True)
            print('Creating model: No mask layers.')
        else:
            # Get the one with masks and pretrained model.
            vgg16 = modnets.vgg16(mask_init, mask_scale, threshold_fn)
            vgg16_pretrained = models.vgg16(pretrained=True)
            # Copy weights from the pretrained to the modified model.
            for module, module_pretrained in zip(vgg16.modules(), vgg16_pretrained.modules()):
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
            print('Creating model: Mask layers created.')

        self.datasets, self.classifiers = [], nn.ModuleList()

        idx = 6
        for module in vgg16.classifier.children():
            if isinstance(module, (nn.Linear, nl.ElementWiseLinear)):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1
        features = list(vgg16.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are common amongst all classes.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(4096, num_outputs))

    def set_dataset(self, dataset):
        """Change the active classifier."""
        assert dataset in self.datasets
        self.classifier = self.classifiers[self.datasets.index(dataset)]

    def train_nobn(self, mode=True):
        """Override the default module train."""
        super(ModifiedVGG16, self).train(mode)

        # Set the BNs to eval mode so that the running means and averages
        # do not update.
        for module in self.shared.modules():
            if 'BatchNorm' in str(type(module)):
                module.eval()

    def forward(self, x):
        x = self.shared(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


class ModifiedVGG16BN(ModifiedVGG16):
    """VGG16 with support for multiple classifiers."""

    def __init__(self, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False):
        super(ModifiedVGG16BN, self).__init__(make_model=False)
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original)

    def make_model(self, mask_init, mask_scale, threshold_fn, original):
        """Creates the model."""
        if original:
            vgg16_bn = models.vgg16_bn(pretrained=True)
            print('Creating model: No mask layers.')
        else:
            # Get the one with masks and pretrained model.
            vgg16_bn = modnets.vgg16_bn(mask_init, mask_scale, threshold_fn)
            vgg16_bn_pretrained = models.vgg16_bn(pretrained=True)
            # Copy weights from the pretrained to the modified model.
            for module, module_pretrained in zip(vgg16_bn.modules(), vgg16_bn_pretrained.modules()):
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    module.running_mean.copy_(module_pretrained.running_mean)
                    module.running_var.copy_(module_pretrained.running_var)
            print('Creating model: Mask layers created.')

        self.datasets, self.classifiers = [], nn.ModuleList()

        idx = 6
        for module in vgg16_bn.classifier.children():
            if isinstance(module, (nn.Linear, nl.ElementWiseLinear)):
                if idx == 6:
                    fc6 = module
                elif idx == 7:
                    fc7 = module
                elif idx == 8:
                    self.datasets.append('imagenet')
                    self.classifiers.append(module)
                idx += 1
        features = list(vgg16_bn.features.children())
        features.extend([
            View(-1, 25088),
            fc6,
            nn.ReLU(inplace=True),
            nn.Dropout(),
            fc7,
            nn.ReLU(inplace=True),
            nn.Dropout(),
        ])

        # Shared params are those which are common amongst all classes.
        self.shared = nn.Sequential(*features)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None


class ModifiedResNet(ModifiedVGG16):
    """ResNet-50."""

    def __init__(self, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False):
        super(ModifiedResNet, self).__init__(make_model=False)
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original)

    def make_model(self, mask_init, mask_scale, threshold_fn, original):
        """Creates the model."""
        if original:
            resnet = models.resnet50(pretrained=True)
            print('Creating model: No mask layers.')
        else:
            # Get the one with masks and pretrained model.
            resnet = modnets.resnet50(mask_init, mask_scale, threshold_fn)
            resnet_pretrained = models.resnet50(pretrained=True)
            # Copy weights from the pretrained to the modified model.
            for module, module_pretrained in zip(resnet.modules(), resnet_pretrained.modules()):
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    if module.bias:
                        module.bias.data.copy_(module_pretrained.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    module.running_mean.copy_(module_pretrained.running_mean)
                    module.running_var.copy_(module_pretrained.running_var)
            print('Creating model: Mask layers created.')

        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        # Add the default imagenet classifier.
        self.datasets.append('imagenet')
        self.classifiers.append(resnet.fc)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(2048, num_outputs))


class ModifiedDenseNet(ModifiedVGG16):
    """DenseNet-121."""

    def __init__(self, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False):
        super(ModifiedDenseNet, self).__init__(make_model=False)
        if make_model:
            self.make_model(mask_init, mask_scale, threshold_fn, original)

    def make_model(self, mask_init, mask_scale, threshold_fn, original):
        """Creates the model."""
        if original:
            densenet = models.densenet121(pretrained=True)
            print('Creating model: No mask layers.')
        else:
            # Get the one with masks and pretrained model.
            densenet = modnets.densenet121(mask_init, mask_scale, threshold_fn)
            densenet_pretrained = models.densenet121(pretrained=True)
            # Copy weights from the pretrained to the modified model.
            for module, module_pretrained in zip(densenet.modules(), densenet_pretrained.modules()):
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    if module.bias:
                        module.bias.data.copy_(module_pretrained.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    module.running_mean.copy_(module_pretrained.running_mean)
                    module.running_var.copy_(module_pretrained.running_var)
            print('Creating model: Mask layers created.')

        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = densenet.features

        # Add the default imagenet classifier.
        self.datasets.append('imagenet')
        self.classifiers.append(densenet.classifier)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None

    def forward(self, x):
        features = self.shared(x)
        out = F.relu(features, inplace=True)
        out = F.avg_pool2d(out, kernel_size=7).view(features.size(0), -1)
        out = self.classifier(out)
        return out

    def add_dataset(self, dataset, num_outputs):
        """Adds a new dataset to the classifier."""
        if dataset not in self.datasets:
            self.datasets.append(dataset)
            self.classifiers.append(nn.Linear(1024, num_outputs))


class ResNetDiffInit(ModifiedResNet):
    """ResNet50 with non-ImageNet initialization."""

    def __init__(self, source, mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer',
                 make_model=True, original=False):
        super(ResNetDiffInit, self).__init__(make_model=False)
        if make_model:
            self.make_model(source, mask_init, mask_scale,
                            threshold_fn, original)

    def make_model(self, source, mask_init, mask_scale, threshold_fn, original):
        """Creates the model."""
        if original:
            resnet = torch.load(source)
            print('Loading model:', source)
        else:
            # Get the one with masks and pretrained model.
            resnet = modnets.resnet50(mask_init, mask_scale, threshold_fn)
            resnet_pretrained = torch.load(source)
            # Copy weights from the pretrained to the modified model.
            for module, module_pretrained in zip(resnet.modules(), resnet_pretrained.modules()):
                if 'ElementWise' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    if module.bias:
                        module.bias.data.copy_(module_pretrained.bias.data)
                elif 'BatchNorm' in str(type(module)):
                    module.weight.data.copy_(module_pretrained.weight.data)
                    module.bias.data.copy_(module_pretrained.bias.data)
                    module.running_mean.copy_(module_pretrained.running_mean)
                    module.running_var.copy_(module_pretrained.running_var)
            print('Creating model: Mask layers created.')

        self.datasets, self.classifiers = [], nn.ModuleList()

        # Create the shared feature generator.
        self.shared = nn.Sequential()
        for name, module in resnet.named_children():
            if name != 'fc':
                self.shared.add_module(name, module)

        # Add the default classifier.
        if 'places' in source:
            self.datasets.append('places')
        elif 'imagenet' in source:
            self.datasets.append('imagenet')
        if original:
            self.classifiers.append(resnet.fc)
        else:
            self.classifiers.append(resnet_pretrained.fc)

        # model.set_dataset() has to be called explicity, else model won't work.
        self.classifier = None
