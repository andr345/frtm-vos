import os
import sys
from pathlib import Path
from easydict import EasyDict as edict
import argparse

p = str(Path(__file__).absolute().parents[2])
if p not in sys.path:
    sys.path.append(p)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # Match nvidia-smi and CUDA device ids

import torch

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

from lib.datasets import DAVISDataset, YouTubeVOSDataset
from model.tracker import Tracker
from model.augmenter import ImageAugmenter
from model.feature_extractor import ResnetFeatureExtractor
from model.seg_network import SegNetwork
from lib.evaluation import evaluate_dataset


class Parameters:

    def __init__(self, weights, fast=False, device="cuda:0"):

        self.device = device
        self.weights = weights
        self.num_aug = 5
        self.train_skipping = 8
        self.learning_rate = 0.1

        # Autodetect the feature extractor

        self.in_channels = weights['refiner.TSE.layer4.reduce.0.weight'].shape[1]
        if self.in_channels == 1024:
            self.feature_extractor = "resnet101"
        elif self.in_channels == 256:
            self.feature_extractor = "resnet18"
        else:
            raise ValueError

        if fast:
            self.init_iters = (5, 10, 10, 10)
            self.update_iters = (5,)
        else:
            self.init_iters = (5, 10, 10, 10, 10)
            self.update_iters = (10,)

        self.aug_params = edict(

            num_aug=self.num_aug,
            min_px_count=1,

            fg_aug_params=edict(
                rotation=[5, -5, 10, -10, 20, -20, 30, -30, 45, -45],
                fliplr=[False, False, False, False, True],
                scale=[0.5, 0.7, 1.0, 1.5, 2.0, 2.5],
                skew=[(0.0, 0.0), (0.0, 0.0), (0.1, 0.1)],
                blur_size=[0.0, 0.0, 0.0, 2.0],
                blur_angle=[0, 45, 90, 135],
            ),
            bg_aug_params=edict(
                tcenter=[(0.5, 0.5)],
                rotation=[0, 0, 0],
                fliplr=[False],
                scale=[1.0, 1.0, 1.2],
                skew=[(0.0, 0.0)],
                blur_size=[0.0, 0.0, 1.0, 2.0, 5.0],
                blur_angle=[0, 45, 90, 135],
            ),
        )

        self.disc_params = edict(
            layer="layer4", in_channels=self.in_channels, c_channels=96, out_channels=1,
            init_iters=self.init_iters, update_iters=self.update_iters,
            memory_size=80, train_skipping=self.train_skipping, learning_rate=self.learning_rate,
            pixel_weighting=dict(method='hinge', tf=0.1),
            filter_reg=(1e-4, 1e-2), precond=(1e-4, 1e-2), precond_lr=0.1, CG_forgetting_rate=750,
            device=self.device, update_filters=True,
        )

        self.refnet_params = edict(
            layers=("layer5", "layer4", "layer3", "layer2"),
            nchannels=64, use_batch_norm=True,
        )

    def get_model(self):

        augmenter = ImageAugmenter(self.aug_params)
        extractor = ResnetFeatureExtractor(self.feature_extractor).to(self.device)
        self.disc_params.in_channels = extractor.get_out_channels()[self.disc_params.layer]

        p = self.refnet_params
        refinement_layers_channels = {L: nch for L, nch in extractor.get_out_channels().items() if L in p.layers}
        refiner = SegNetwork(self.disc_params.out_channels, p.nchannels, refinement_layers_channels, p.use_batch_norm)

        mdl = Tracker(augmenter, extractor, self.disc_params, refiner, self.device)
        mdl.load_state_dict(self.weights)
        mdl.to(self.device)

        return mdl


if __name__ == '__main__':

    # Edit these paths for your local setup

    paths = dict(
        models=Path(__file__).parent / "weights",  # The .pth files should be here
        davis="/path/to/DAVIS",  # DAVIS dataset root
        yt2018="/path/to/ytvos2018",  # YouTubeVOS 2018 root
        output="/path/to/results",  # Output path
    )
    # paths = dict(
    #     models=Path(__file__).parent / "weights",  # The .pth files should be here
    #     davis="/mnt/data/camrdy_ws/DAVIS",  # DAVIS dataset root
    #     yt2018="/mnt/data/camrdy_ws/YouTubeVOS/2018",  # YouTubeVOS 2018 root
    #     output="/mnt/data/camrdy_ws/results",  # Output path
    # )

    datasets = dict(dv2016val=(DAVISDataset, dict(path=paths['davis'], year='2016', split='val')),
                    dv2017val=(DAVISDataset, dict(path=paths['davis'], year='2017', split='val')),
                    yt2018jjval=(YouTubeVOSDataset, dict(path=paths['yt2018'], year='2018', split='jjval_all_frames')),
                    yt2018val=(YouTubeVOSDataset, dict(path=paths['yt2018'], year='2018', split='valid_all_frames')))

    args_parser = argparse.ArgumentParser(description='Evaluate FRTM on a validation dataset')
    args_parser.add_argument('--model', type=str, help='name of model weights file', required=True)
    args_parser.add_argument('--dset', type=str, required=True, choices=datasets.keys(), help='Dataset name.')
    args_parser.add_argument('--dev', type=str, default="cuda:0", help='Target device to run on.')
    args_parser.add_argument('--fast', action='store_true', default=False, help='Whether to use fewer optimizer steps.')

    args = args_parser.parse_args()

    # Setup

    model_path = Path(paths['models']).expanduser().resolve() / args.model
    if not model_path.exists():
        print("Model file '%s' not found." % model_path)
        quit(1)
    weights = torch.load(model_path, map_location='cpu')['model']

    dset = datasets[args.dset]
    dset = dset[0](**dset[1])

    ex_name = dset.name + "-" + model_path.stem + ("_fast" if args.fast else "")
    out_path = Path(paths['output']).expanduser().resolve() / ex_name
    out_path.mkdir(exist_ok=True, parents=True)

    # Run the tracker and evaluate the results

    p = Parameters(weights)
    tracker = p.get_model()
    tracker.run_dataset(dset, out_path, speedrun=args.dset == 'dv2016val')

    dset.all_annotations = True
    print()
    print("Computing J-scores")
    evaluate_dataset(dset, out_path, measure='J')
    print()
    print("Computing F-scores")
    evaluate_dataset(dset, out_path, measure='F')
