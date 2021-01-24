from pathlib import Path
from easydict import EasyDict as edict
import torch

if __name__ == '__main__':
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

from ytvos_validation.ytvos_dataset import YouTubeVOSTestDataset
from ytvos_validation.utils import ModuleWrapper
from ytvos_validation.augmenter import ImageAugmenter
from ytvos_validation.feature_extractor import FeatureExtractor
from ytvos_validation.seg_network import SegNetwork
from ytvos_validation.tracker import Tracker


class Parameters:

    def __init__(self):

        # Paths

        self.device = 'cuda:0'

        # Model parameters

        self.feature_extractor = 'resnet101'

        self.train_skipping = 8
        self.cdims = 96
        self.init_iters = [5, 10, 10, 10, 10]
        self.num_aug = 5
        self.update_iters = [10]
        self.update_filters = True

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

            device=self.device,
            sample_memory_size=80,
            debug=0,
            update_filters=self.update_filters,
            train_skipping=self.train_skipping,

            # Per-feature layer parameters

            pixel_weighting=dict(method='hinge', tf=0.1, distractor_mult=1.0, per_frame=True,
                                 update_method="thresh", max_fg_weight=100),
            layer="layer4",
            cdims=self.cdims,
            out_channels=1,
            learning_rate=0.1,
            kernel_size=[3],
            filter_reg=[1e-4, 1e-2],
            precon=[1e-4, 1e-2],
            n_channels=[1],
            with_bias=False,
            init_iters=self.init_iters,
            update_iters=self.update_iters,

            # Conjugate Gradient parameters

            fletcher_reeves=False,  # Use the Fletcher-Reeves (true) or Polak-Ribiere (false) formula in the CG
            standard_alpha=True,    # Use the standard formula for computing the step length in Conjugate Gradient
            CG_forgetting_rate=75,  # Forgetting rate of the last conjugate direction

            clamp_output=False,
            compute_norm=False,
        )

        self.refnet_params = edict(
            refinement_layers=["layer5", "layer4", "layer3", "layer2"],
            nchannels=64, use_batch_norm=True,
        )

    def get_model(self, mdl_path):

        augmenter = ImageAugmenter(self.aug_params)
        extractor = FeatureExtractor(device=self.device, name=self.feature_extractor)
        extractor = ModuleWrapper(extractor)

        layers_channels = extractor.get_out_channels()
        p = self.refnet_params
        refinement_layers_channels = {L: nch for L, nch in layers_channels.items() if L in p.refinement_layers}
        refiner = SegNetwork(self.disc_params.out_channels, p.nchannels, refinement_layers_channels, p.use_batch_norm)

        mdl = Tracker(augmenter, extractor, self.disc_params, refiner, self.device)
        checkpoint = torch.load(str(mdl_path), map_location='cpu')
        mdl.load_state_dict(checkpoint['model'])

        return mdl.to(self.device)


if __name__ == '__main__':

    dset_path = Path("/mnt/data/datasets/YouTubeVOS/")
    log_path = Path.home() / "workspace" / Path(__file__).stem / "Annotations"
    mdl_path = Path.home() / "workspace/models/lfrtm_rn101_ytvos_ep249.pth"

    params = Parameters()
    tracker = params.get_model(mdl_path)

    dset = YouTubeVOSTestDataset(dset_path, name="valid_all_frames", year=2018, sequences=None)
    tracker.run_dataset(dset, log_path)
