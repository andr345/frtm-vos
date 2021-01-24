import torch
from torch import nn
from torch.nn import functional as F
from .utils import conv, relu, interpolate, adaptive_cat


class TSE(nn.Module):

    def __init__(self, fc, ic, oc):
        super().__init__()

        nc = ic + oc
        self.reduce = nn.Sequential(conv(fc, oc, 1), relu(), conv(oc, oc, 1))
        self.transform = nn.Sequential(conv(nc, nc, 3), relu(),
                                       conv(nc, nc, 3), relu(),
                                       conv(nc, oc, 3), relu())

    def forward(self, ft, score, x=None):

        h = self.reduce(ft)
        hpool = F.adaptive_avg_pool2d(h, (1, 1)) if x is None else x
        h = adaptive_cat((h, score), dim=1, ref_tensor=0)
        h = self.transform(h)
        return h, hpool


class CAB(nn.Module):

    def __init__(self, oc, deepest):
        super().__init__()

        self.convreluconv = nn.Sequential(conv(2 * oc, oc, 1), relu(), conv(oc, oc, 1))
        self.deepest = deepest

    def forward(self, deeper, shallower):

        shallow_pool = F.adaptive_avg_pool2d(shallower, (1, 1))
        deeper_pool = deeper if self.deepest else F.adaptive_avg_pool2d(deeper, (1, 1))
        global_pool = torch.cat((shallow_pool, deeper_pool), dim=1)
        conv_1x1 = self.convreluconv(global_pool)
        inputs = shallower * torch.sigmoid(conv_1x1)
        out = inputs + interpolate(deeper, inputs.shape[-2:])

        return out


class RRB(nn.Module):

    def __init__(self, oc, use_bn=False):
        super().__init__()
        self.conv1x1 = conv(oc, oc, 1)
        if use_bn:
            self.bblock = nn.Sequential(conv(oc, oc, 3), nn.BatchNorm2d(oc), relu(), conv(oc, oc, 3, bias=False))
        else:
            self.bblock = nn.Sequential(conv(oc, oc, 3), relu(), conv(oc, oc, 3, bias=False))  # Basic block

    def forward(self, x):
        h = self.conv1x1(x)
        return F.relu(h + self.bblock(h))


class Upsampler(nn.Module):

    def __init__(self, in_channels=64):
        super().__init__()

        self.conv1 = conv(in_channels, in_channels//2, 3)
        self.conv2 = conv(in_channels//2, 1, 3)

    def forward(self, x, image_size):
        x = F.interpolate(x, (2*x.shape[-2], 2*x.shape[-1]), mode='bicubic', align_corners=False)
        x = F.relu(self.conv1(x))
        x = F.interpolate(x, image_size[-2:], mode='bicubic', align_corners=False)
        x = self.conv2(x)
        return x


class SegNetwork(nn.Module):

    def __init__(self, in_channels=1, out_channels=32, ft_channels=None, use_bn=False):

        super().__init__()

        assert ft_channels is not None
        self.ft_channels = ft_channels

        self.TSE = nn.ModuleDict()
        self.RRB1 = nn.ModuleDict()
        self.CAB = nn.ModuleDict()
        self.RRB2 = nn.ModuleDict()

        ic = in_channels
        oc = out_channels

        for L, fc in self.ft_channels.items():
            self.TSE[L] = TSE(fc, ic, oc)
            self.RRB1[L] = RRB(oc, use_bn=use_bn)
            self.CAB[L] = CAB(oc, L == 'layer5')
            self.RRB2[L] = RRB(oc, use_bn=use_bn)

        self.project = Upsampler(out_channels)

    def forward(self, scores, features, image_size):

        x = None
        for i, L in enumerate(self.ft_channels):
            ft = features[L]
            s = [interpolate(ss, ft.shape[-2:]) for ss in scores]  # Resample scores to match features size
            s = torch.cat(s, dim=1)

            h, hpool = self.TSE[L](ft, s, x)

            h = self.RRB1[L](h)
            h = self.CAB[L](hpool, h)
            x = self.RRB2[L](h)

        x = self.project(x, image_size)
        return x
