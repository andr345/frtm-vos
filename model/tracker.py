import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import tqdm

from lib.image import imwrite_indexed
from lib.utils import AverageMeter
from .augmenter import ImageAugmenter
from .discriminator import Discriminator
from .seg_network import SegNetwork

from time import time


class TargetObject:

    def __init__(self, obj_id, disc_params, **kwargs):

        self.object_id = obj_id
        self.discriminator = Discriminator(**disc_params)
        self.disc_layer = disc_params.layer
        self.start_frame = None
        self.start_mask = None
        self.index = -1

        for key, val in kwargs.items():
            setattr(self, key, val)

    def initialize(self, ft, mask):
        self.discriminator.init(ft[self.disc_layer], mask)

    def classify(self, ft):
        return self.discriminator.apply(ft)


class Tracker(nn.Module):

    def __init__(self, augmenter: ImageAugmenter, feature_extractor, disc_params, refiner: SegNetwork, device):

        super().__init__()

        self.augmenter = augmenter
        self.augment = augmenter.augment_first_frame
        self.disc_params = disc_params
        self.feature_extractor = feature_extractor

        self.refiner = refiner
        for m in self.refiner.parameters():
            m.requires_grad_(False)
        self.refiner.eval()

        self.device = device

        self.first_frames = []
        self.current_frame = 0
        self.current_masks = None
        self.num_objects = 0

    def clear(self):
        self.first_frames = []
        self.current_frame = 0
        self.current_masks = None
        self.num_objects = 0
        torch.cuda.empty_cache()


    def run_dataset(self, dataset, out_path, speedrun=False, restart=None):
        """
        :param dataset:   Dataset to work with (See datasets.py)
        :param out_path:  Root path for storing label images. Sequences of label pngs will be created in subdirectories.
        :param speedrun:  [Optional] Whether or not to warm up Pytorch when measuring the run time. Default: False
        :param restart:   [Optional] Name of sequence to restart from. Useful for debugging. Default: None
        """
        out_path.mkdir(exist_ok=True, parents=True)

        dset_fps = AverageMeter()

        print('Evaluating', dataset.name)

        restarted = False
        for sequence in dataset:
            if restart is not None and not restarted:
                if sequence.name != restart:
                    continue
                restarted = True

            # We preload data as we cannot both read from disk and upload to the GPU in the background,
            # which would be a reasonable thing to do. However, in PyTorch, it is hard or impossible
            # to upload images to the GPU in a data loader running as a separate process.
            sequence.preload(self.device)
            self.clear()  # Mitigate out-of-memory that may occur on some YouTubeVOS sequences on 11GB devices.
            outputs, seq_fps = self.run_sequence(sequence, speedrun)
            dset_fps.update(seq_fps)

            dst = out_path / sequence.name
            dst.mkdir(exist_ok=True)
            for lb, f in zip(outputs, sequence.frame_names):
                imwrite_indexed(dst / (f + ".png"), lb)

        print("Average frame rate: %.2f fps" % dset_fps.avg)

    def run_sequence(self, sequence, speedrun=False):
        """
        :param sequence:  FileSequence to run.
        :param speedrun:  Only for DAVIS 2016: If True, let pytorch initialize its buffers in advance
                          to not incorrectly measure the memory allocation time in the first frame.
        :return:
        """

        self.eval()
        self.object_ids = sequence.obj_ids
        self.current_frame = 0
        self.targets = dict()

        N = 0

        object_ids = torch.tensor([0] + sequence.obj_ids, dtype=torch.uint8, device=self.device)  # Mask -> labels LUT

        if speedrun:
            image, labels, obj_ids = sequence[0]
            image = image.to(self.device)
            labels = labels.to(self.device)
            self.initialize(image, labels, sequence.obj_ids)  # Assume DAVIS 2016
            self.track(image)
            torch.cuda.synchronize()
            self.targets = dict()

        outputs = []
        t0 = time()
        for i, (image, labels, new_objects) in tqdm.tqdm(enumerate(sequence), desc=sequence.name, total=len(sequence), unit='frames'):

            old_objects = set(self.targets.keys())

            image = image.to(self.device)
            if len(new_objects) > 0:
                labels = labels.to(self.device)
                self.initialize(image, labels, new_objects)

            if len(old_objects) > 0:
                self.track(image)

                masks = self.current_masks
                if len(sequence.obj_ids) == 1:
                    labels = object_ids[(masks[1:2] > 0.5).long()]
                else:
                    masks = torch.clamp(masks, 1e-7, 1 - 1e-7)
                    masks[0:1] = torch.min((1 - masks[1:]), dim=0, keepdim=True)[0]  # background activation
                    segs = F.softmax(masks / (1 - masks), dim=0)  # s = one-hot encoded object activations
                    labels = object_ids[segs.argmax(dim=0)]

            if isinstance(labels, list) and len(labels) == 0:  # No objects yet
                labels = image.new_zeros(1, *image.shape[-2:])

            outputs.append(labels)
            self.current_frame += 1
            N += 1

        torch.cuda.synchronize()
        T = time() - t0
        fps = N / T

        return outputs, fps

    def initialize(self, image, labels, new_objects):

        self.current_masks = torch.zeros((len(self.targets) + len(new_objects) + 1, *image.shape[-2:]), device=self.device)

        for obj_id in new_objects:

            # Create target

            mask = (labels == obj_id).byte()
            target = TargetObject(obj_id=obj_id, index=len(self.targets)+1, disc_params=self.disc_params,
                                  start_frame=self.current_frame, start_mask=mask)
            self.targets[obj_id] = target

            # HACK for debugging
            torch.random.manual_seed(0)
            np.random.seed(0)

            # Augment first image and extract features

            im, msk = self.augment(image, mask)
            with torch.no_grad():
                ft = self.feature_extractor(im, [target.disc_layer])
            target.initialize(ft, msk)

            self.current_masks[target.index] = mask

        return self.current_masks

    def track(self, image):

        im_size = image.shape[-2:]
        features = self.feature_extractor(image)

        # Classify

        for obj_id, target in self.targets.items():
            if target.start_frame < self.current_frame:
                s = target.classify(features[target.disc_layer])
                y = torch.sigmoid(self.refiner(s, features, im_size))
                self.current_masks[target.index] = y

        # Update

        for obj_id, t1 in self.targets.items():
            if t1.start_frame < self.current_frame:
                for obj_id2, t2 in self.targets.items():
                    if obj_id != obj_id2 and t2.start_frame == self.current_frame:
                        self.current_masks[t1.index] *= (1 - t2.start_mask.squeeze(0)).float()

        p = torch.clamp(self.current_masks, 1e-7, 1 - 1e-7)
        p[0:1] = torch.min((1 - p[1:]), dim=0, keepdim=True)[0]
        segs = F.softmax(p / (1 - p), dim=0)
        inds = segs.argmax(dim=0)

        # self.out_buffer = segs * F.one_hot(inds, segs.shape[0]).permute(2, 0, 1)
        for i in range(self.current_masks.shape[0]):
            self.current_masks[i] = segs[i] * (inds == i).float()

        for obj_id, target in self.targets.items():
            if target.start_frame < self.current_frame and self.disc_params.update_filters:
                target.discriminator.update(self.current_masks[target.index].unsqueeze(0).unsqueeze(0))

        return self.current_masks
