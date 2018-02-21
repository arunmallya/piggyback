import torch.nn as nn
import torch.utils.model_zoo as model_zoo
import math

import modnets.layers as nl

__all__ = [
    'VGG', 'vgg16', 'vgg16_bn'
]


class VGG(nn.Module):

    def __init__(self, features, mask_init, mask_scale, threshold_fn, num_classes=1000):
        super(VGG, self).__init__()
        self.features = features
        self.classifier = nn.Sequential(
            nl.ElementWiseLinear(
                512 * 7 * 7, 4096, mask_init=mask_init, mask_scale=mask_scale,
                threshold_fn=threshold_fn),
            nn.ReLU(True),
            nn.Dropout(),
            nl.ElementWiseLinear(
                4096, 4096, mask_init=mask_init, mask_scale=mask_scale,
                threshold_fn=threshold_fn),
            nn.ReLU(True),
            nn.Dropout(),
            nn.Linear(4096, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x


def make_layers(cfg, mask_init, mask_scale, threshold_fn, batch_norm=False):
    layers = []
    in_channels = 3
    for v in cfg:
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nl.ElementWiseConv2d(
                in_channels, v, kernel_size=3, padding=1,
                mask_init=mask_init, mask_scale=mask_scale,
                threshold_fn=threshold_fn)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
            else:
                layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


cfg = {
    'A': [64, 'M', 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'B': [64, 64, 'M', 128, 128, 'M', 256, 256, 'M', 512, 512, 'M', 512, 512, 'M'],
    'D': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M'],
    'E': [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 256, 'M', 512, 512, 512, 512, 'M', 512, 512, 512, 512, 'M'],
}


def vgg16(mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', **kwargs):
    """VGG 16-layer model (configuration "D")."""
    model = VGG(make_layers(cfg['D'], mask_init, mask_scale, threshold_fn),
                mask_init, mask_scale, threshold_fn, **kwargs)
    return model


def vgg16_bn(mask_init='1s', mask_scale=1e-2, threshold_fn='binarizer', **kwargs):
    """VGG 16-layer model (configuration "D") with batch normalization."""
    model = VGG(make_layers(cfg['D'], mask_init, mask_scale, threshold_fn, batch_norm=True),
                mask_init, mask_scale, threshold_fn, **kwargs)
    return model
