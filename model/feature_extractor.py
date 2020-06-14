from collections import OrderedDict as odict
import torch
from torchvision.models import resnet18, resnet34, resnet50, resnet101
from lib.utils import get_out_channels


class ResnetFeatureExtractor:

    def __init__(self, name='resnet101'):
        super().__init__()

        networks = {"resnet18": resnet18, "resnet34": resnet34, "resnet50": resnet50, "resnet101": resnet101}

        self.resnet = networks[name](pretrained=True)
        del self.resnet.avgpool, self.resnet.fc
        for m in self.resnet.parameters():
            m.requires_grad = False
        self.resnet.eval()

        self._out_channels = odict(  # Deep-to-shallow order is required by SegNetwork
            layer5=get_out_channels(self.resnet.layer4),
            layer4=get_out_channels(self.resnet.layer3),
            layer3=get_out_channels(self.resnet.layer2),
            layer2=get_out_channels(self.resnet.layer1),
            layer1=get_out_channels(self.resnet.conv1))

        maxval = 255
        stds = torch.tensor((0.229, 0.224, 0.225), dtype=torch.float, requires_grad=False).reshape(1, 3, 1, 1)
        means = torch.tensor((0.485, 0.456, 0.406), dtype=torch.float, requires_grad=False).reshape(1, 3, 1, 1)

        self.norm_weight = (1 / maxval / stds)
        self.norm_bias = (-means / stds)

    def to(self, device):
        self.resnet.to(device)
        self.norm_weight = self.norm_weight.to(device)
        self.norm_bias = self.norm_bias.to(device)
        return self

    def __call__(self, input, output_layers=None):

        x = self.norm_weight * input.float() + self.norm_bias

        out = dict()

        def save_out(L, x):
            if output_layers is None or L in output_layers:
                out[L] = x

        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        save_out('layer1', x)

        x = self.resnet.layer1(x)
        save_out('layer2', x)

        x = self.resnet.layer2(x)
        save_out('layer3', x)

        x = self.resnet.layer3(x)
        save_out('layer4', x)

        x = self.resnet.layer4(x)
        save_out('layer5', x)

        return out

    def get_out_channels(self):
        return self._out_channels

    def no_grad_forward(self, input, output_layers=None, chunk_size=None):
        """
        :param input:
        :param output_layers:  List of layer names (layer1 ...) to keep
        :param chunk_size:  [Optional] Split the batch into chunks of this size
                            and process them sequentially to save memory.
        :return: dict of output tensors
        """

        with torch.no_grad():
            if chunk_size is None:
                return self(input, output_layers)
            else:
                outputs = [self(t, output_layers) for t in torch.split(input, chunk_size)]
                return {L: torch.cat([out[L] for out in outputs]) for L in outputs[0]}