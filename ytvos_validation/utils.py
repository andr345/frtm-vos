import torch
from torch import nn as nn
from torch.nn import functional as F


class ModuleWrapper:
    """ A wrapper for hiding modules from PyTorch, so that the same module can be used in multiple places.
    and yet saved only once in per checkpoint, or not at all. """

    # https://stackoverflow.com/questions/1466676/create-a-wrapper-class-to-call-a-pre-and-post-function-around-existing-functions

    def __init__(self, wrapped_module):
        self.__wrapped_module__ = wrapped_module

    def __getattr__(self, attr):
        orig_attr = self.__wrapped_module__.__getattribute__(attr)
        if callable(orig_attr):
            def hooked(*args, **kwargs):
                result = orig_attr(*args, **kwargs)
                # prevent wrapped_class from becoming unwrapped
                if result == self.__wrapped_module__:
                    return self
                return result

            return hooked
        else:
            return orig_attr

    def __call__(self, *args, **kwargs):
        return self.__wrapped_module__(*args, **kwargs)


def conv(ic, oc, ksize, bias=True, dilation=1, stride=1):
    return nn.Conv2d(ic, oc, ksize, padding=ksize // 2, bias=bias, dilation=dilation, stride=stride)


def relu(negative_slope=0.0, inplace=False):
    return nn.LeakyReLU(negative_slope, inplace=inplace)


def interpolate(t, sz):
    sz = sz.tolist() if torch.is_tensor(sz) else sz
    return F.interpolate(t, sz, mode='bilinear', align_corners=False) if t.shape[-2:] != sz else t


def adaptive_cat(seq, dim=0, ref_tensor=0):
    sz = seq[ref_tensor].shape[-2:]
    t = torch.cat([interpolate(t, sz) for t in seq], dim=dim)
    return t


def get_out_channels(layer):

    if hasattr(layer, 'out_channels'):
        oc = layer.out_channels
    elif hasattr(layer, '_modules'):
        oc = get_out_channels(layer._modules)
    else:
        ocs = []
        for key in reversed(layer):
            ocs.append(get_out_channels(layer[key]))

        oc = 0
        for elem in ocs:
            if elem:
                return elem

    return oc
