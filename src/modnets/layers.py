"""Contains novel layer definitions."""
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from torch.nn.modules.utils import _pair
from torch.nn.parameter import Parameter

DEFAULT_THRESHOLD = 5e-3


class Binarizer(torch.autograd.Function):
    """Binarizes {0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(Binarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs[inputs.le(self.threshold)] = 0
        outputs[inputs.gt(self.threshold)] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput


class Ternarizer(torch.autograd.Function):
    """Ternarizes {-1, 0, 1} a real valued tensor."""

    def __init__(self, threshold=DEFAULT_THRESHOLD):
        super(Ternarizer, self).__init__()
        self.threshold = threshold

    def forward(self, inputs):
        outputs = inputs.clone()
        outputs.fill_(0)
        outputs[inputs < 0] = -1
        outputs[inputs > self.threshold] = 1
        return outputs

    def backward(self, gradOutput):
        return gradOutput


class ElementWiseConv2d(nn.Module):
    """Modified conv with masks for weights."""

    def __init__(self, in_channels, out_channels, kernel_size, stride=1,
                 padding=0, dilation=1, groups=1, bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None):
        super(ElementWiseConv2d, self).__init__()
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        if in_channels % groups != 0:
            raise ValueError('in_channels must be divisible by groups')
        if out_channels % groups != 0:
            raise ValueError('out_channels must be divisible by groups')
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation
        self.transposed = False
        self.output_padding = _pair(0)
        self.groups = groups

        # weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(
            out_channels, in_channels // groups, *kernel_size), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(
                out_channels), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # Initialize real-valued mask weights.
        self.mask_real = self.weight.data.new(self.weight.size())
        if mask_init == '1s':
            self.mask_real.fill_(mask_scale)
        elif mask_init == 'uniform':
            self.mask_real.uniform_(-1 * mask_scale, mask_scale)
        # mask_real is now a trainable parameter.
        self.mask_real = Parameter(self.mask_real)

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            print('Calling binarizer with threshold:', threshold)
            self.threshold_fn = Binarizer(threshold=threshold)
        elif threshold_fn == 'ternarizer':
            print('Calling ternarizer with threshold:', threshold)
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input):
        # Get binarized/ternarized mask from real-valued mask.
        mask_thresholded = self.threshold_fn(self.mask_real)
        # Mask weights with above mask.
        weight_thresholded = mask_thresholded * self.weight
        # Perform conv using modified weight.
        return F.conv2d(input, weight_thresholded, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)

    def __repr__(self):
        s = ('{name} ({in_channels}, {out_channels}, kernel_size={kernel_size}'
             ', stride={stride}')
        if self.padding != (0,) * len(self.padding):
            s += ', padding={padding}'
        if self.dilation != (1,) * len(self.dilation):
            s += ', dilation={dilation}'
        if self.output_padding != (0,) * len(self.output_padding):
            s += ', output_padding={output_padding}'
        if self.groups != 1:
            s += ', groups={groups}'
        if self.bias is None:
            s += ', bias=False'
        s += ')'
        return s.format(name=self.__class__.__name__, **self.__dict__)

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        if self.bias is not None and self.bias.data is not None:
            self.bias.data = fn(self.bias.data)


class ElementWiseLinear(nn.Module):
    """Modified linear layer."""

    def __init__(self, in_features, out_features, bias=True,
                 mask_init='1s', mask_scale=1e-2,
                 threshold_fn='binarizer', threshold=None):
        super(ElementWiseLinear, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.threshold_fn = threshold_fn
        self.mask_scale = mask_scale
        self.mask_init = mask_init

        if threshold is None:
            threshold = DEFAULT_THRESHOLD
        self.info = {
            'threshold_fn': threshold_fn,
            'threshold': threshold,
        }

        # weight and bias are no longer Parameters.
        self.weight = Variable(torch.Tensor(
            out_features, in_features), requires_grad=False)
        if bias:
            self.bias = Variable(torch.Tensor(
                out_features), requires_grad=False)
        else:
            self.register_parameter('bias', None)

        # Initialize real-valued mask weights.
        self.mask_real = self.weight.data.new(self.weight.size())
        if mask_init == '1s':
            self.mask_real.fill_(mask_scale)
        elif mask_init == 'uniform':
            self.mask_real.uniform_(-1 * mask_scale, mask_scale)
        # mask_real is now a trainable parameter.
        self.mask_real = Parameter(self.mask_real)

        # Initialize the thresholder.
        if threshold_fn == 'binarizer':
            self.threshold_fn = Binarizer(threshold=threshold)
        elif threshold_fn == 'ternarizer':
            self.threshold_fn = Ternarizer(threshold=threshold)

    def forward(self, input):
        # Get binarized/ternarized mask from real-valued mask.
        mask_thresholded = self.threshold_fn(self.mask_real)
        # Mask weights with above mask.
        weight_thresholded = mask_thresholded * self.weight
        # Get output using modified weight.
        return F.linear(input, weight_thresholded, self.bias)

    def __repr__(self):
        return self.__class__.__name__ + '(' \
            + 'in_features=' + str(self.in_features) \
            + ', out_features=' + str(self.out_features) + ')'

    def _apply(self, fn):
        for module in self.children():
            module._apply(fn)

        for param in self._parameters.values():
            if param is not None:
                # Variables stored in modules are graph leaves, and we don't
                # want to create copy nodes, so we have to unpack the data.
                param.data = fn(param.data)
                if param._grad is not None:
                    param._grad.data = fn(param._grad.data)

        for key, buf in self._buffers.items():
            if buf is not None:
                self._buffers[key] = fn(buf)

        self.weight.data = fn(self.weight.data)
        self.bias.data = fn(self.bias.data)
