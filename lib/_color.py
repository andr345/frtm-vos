import torch
import torch.nn as nn
import numpy as np
from PIL import Image

davis_palette = np.repeat(np.expand_dims(np.arange(0,256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                         [0, 64, 128], [128, 64, 128]]


def indexed_to_rgb(im, color_palette=None):
    """
    :param im:  Image, shape (H,W)
    :param color_palette:
    :return:
    """

    if color_palette is None:
        color_palette = davis_palette

    if torch.is_tensor(im):
        p = torch.from_numpy(color_palette).to(im.device)
        im = p[im.squeeze(0).long()].permute(2, 0, 1)
    else:
        im = Image.fromarray(im.squeeze(), 'P')
        im.putpalette(color_palette.ravel())
        im = np.array(im.convert('RGB'))

    return im


class ColorNormalizer(nn.Module):

    def __init__(self, means, stds, maxval, inverse=False):
        super().__init__()

        stds = torch.Tensor(stds).view(3, 1, 1)
        means = torch.Tensor(means).view(3, 1, 1)

        if not inverse:  # Normalize
            gain = 1 / maxval / stds
            bias = -means / stds
        else:  # Denormalize
            gain = stds * maxval
            bias = means * maxval

        self.register_buffer('gain', gain)
        self.register_buffer('bias', bias)

    def forward(self, im):
        """
        :param im:  Image to (de-)normalize, shape=(3,H,W)
        :return: (De-)normalized image
        """
        return im * self.gain + self.bias


class Colormap(nn.Module):

    def __init__(self, values, colors, palette_size):
        super().__init__()

        self.size = palette_size
        self.low = values[0]
        self.high = values[-1]
        self.scale = self.size / (self.high - self.low)

        x = np.arange(self.low, self.high, step=1/self.scale)
        palette = np.stack((np.interp(x=x, xp=values, fp=colors[:, 0]),
                            np.interp(x=x, xp=values, fp=colors[:, 1]),
                            np.interp(x=x, xp=values, fp=colors[:, 2])), axis=-1)
        palette = torch.from_numpy(palette.astype(np.float32))

        self.register_buffer("palette", palette)

    def forward(self, x):
        x = self.palette[torch.round((x - self.low) * self.scale).clamp(0, self.size-1).long()]
        return x.transpose(-1, -2).transpose(-2, -3)


def get_colormap(name, size, low, high):

    if name == 'fire_ice':
        # Based on https://www.mathworks.com/matlabcentral/fileexchange/24870-fireice-hot-cold-colormap
        # Copyright (c) 2009, Joseph Kirk, License: BSD 2-clause
        # Piecewise linear interpolation settings:
        colors = np.array(((0.75, 1, 1), (0, 1, 1), (0, 0, 1), (0, 0, 0), (1, 0, 0), (1, 1, 0), (1, 1, 0.75)))
        values = np.array((-3, -2, -1, 0, 1, 2, 3), dtype=np.float32) / 6 + 0.5  # scaled to [0., 1.]
        values = values * (high - low) + low  # scaled to [low, high]
    else:
        raise ValueError("Undefined colormap %s" % name)

    return Colormap(values, colors, size)


def create_color_normalizer(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), maxval=255):
    """ Get a function object that can denormalize images (mean=0, std=1) """
    return ColorNormalizer(mean, std, maxval, inverse=False)


def create_color_denormalizer(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), maxval=255):
    """ Get a function object that can denormalize images """
    return ColorNormalizer(mean, std, maxval, inverse=True)
