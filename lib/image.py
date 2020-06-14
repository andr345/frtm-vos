import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
import cv2
from ._npp import nppig_cpp

davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                         [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                         [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                         [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                         [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                         [0, 64, 128], [128, 64, 128]]


def imread(filename):
    im = np.array(Image.open(filename))
    im = np.ascontiguousarray(np.atleast_3d(im).transpose(2, 0, 1))
    im = torch.from_numpy(im)
    return im


def imwrite(filename, im):
    assert len(im.shape) < 4 or im.shape[0] == 1  # requires batch size 1
    im = im.detach().cpu().view(-1, *im.shape[-2:]).permute(1, 2, 0).numpy()
    Image.fromarray(im).save(filename)


def imwrite_indexed(filename, im, color_palette=None):
    assert len(im.shape) < 4 or im.shape[0] == 1  # requires batch size 1
    color_palette = davis_palette if color_palette is None else color_palette
    im = Image.fromarray(im.detach().cpu().squeeze().numpy(), 'P')
    im.putpalette(color_palette.ravel())
    im.save(filename)


def warp_affine(src, H, size, mode='bicubic'):
    assert len(src.shape) < 4 or src.shape[0] == 1  # requires batch size 1

    no_cdim = len(src.shape) == 2
    src = src.view(-1, *src.shape[-2:])
    dst = src.new_zeros(src.shape[0], *size)
    H = H.astype(np.float32)[:2, :]

    if src.device.type == 'cpu':
        mode = dict(nearest=cv2.INTER_NEAREST, bilinear=cv2.INTER_LINEAR, bicubic=cv2.INTER_CUBIC,
                    area=cv2.INTER_AREA)[mode]
        for c in range(src.shape[0]):
            cv2.warpAffine(src[c, ...].numpy(), H, (size[1], size[0]), dst[c, ...].numpy(), mode)

    elif src.device.type.startswith('cuda'):
        nppig_cpp.warp_affine(src, dst, torch.from_numpy(H), mode)

    else:
        raise NotImplementedError

    dst = dst.squeeze(0) if no_cdim else dst
    return dst
