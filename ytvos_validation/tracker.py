from pathlib import Path
from tqdm import tqdm
from PIL import Image
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from .ytvos_dataset import Sequence
from .utils import interpolate
from .augmenter import ImageAugmenter
from .discriminator import Discriminator


class TargetObject:

    def __init__(self, **kwargs):

        self.discriminator = None
        self.start_frame = 0
        self.start_mask = None

        for key, val in kwargs.items():
            setattr(self, key, val)


class Tracker(nn.Module):

    def __init__(self, augmenter: ImageAugmenter, feature_extractor, disc_params, refiner, device):

        super().__init__()

        self.augmenter = augmenter
        self.augment = augmenter.augment_first_frame
        self.disc_params = disc_params
        self.feature_extractor = feature_extractor

        self.refiner = refiner
        self.current_frame = 0
        self.num_objects = 0
        self.device = device

        self.davis_palette = np.repeat(np.expand_dims(np.arange(0, 256), 1), 3, 1).astype(np.uint8)
        self.davis_palette[:22, :] = [[0, 0, 0], [128, 0, 0], [0, 128, 0], [128, 128, 0],
                                      [0, 0, 128], [128, 0, 128], [0, 128, 128], [128, 128, 128],
                                      [64, 0, 0], [191, 0, 0], [64, 128, 0], [191, 128, 0],
                                      [64, 0, 128], [191, 0, 128], [64, 128, 128], [191, 128, 128],
                                      [0, 64, 0], [128, 64, 0], [0, 191, 0], [128, 191, 0],
                                      [0, 64, 128], [128, 64, 128]]

    @staticmethod
    def merge_segmentations(fg, obj_ids):

        fg = torch.clamp(fg, 1e-7, 1 - 1e-7)
        bg = torch.min((1 - fg), dim=0, keepdim=True)[0]
        p = torch.cat((bg, fg), dim=0)
        segs = F.softmax(p / (1 - p), dim=0)  # s = one-hot encoded object activations

        obj_ids = torch.tensor([0] + obj_ids)  # vector of object ids, 0 = background
        return segs, obj_ids

    def imwrite_indexed(self, filename, array):

        array = array.detach().cpu().numpy()
        im = Image.fromarray(array.squeeze(), 'P')
        im.putpalette(self.davis_palette.ravel())
        im.save(filename)

    def run_dataset(self, dset, log_path):

        loader = DataLoader(dset, num_workers=0)
        n_seqs = len(loader)

        for j, sequence in enumerate(loader):
            seq, out_labels = self.run_sequence(sequence, j, n_seqs)

            seq_path = log_path / seq.name
            seq_path.mkdir(exist_ok=True, parents=True)
            for k, fid in enumerate(seq.frames):
                self.imwrite_indexed(seq_path / f"{fid}.png", out_labels[k].byte())

    def run_sequence(self, sequence, j, n_seqs, out_device='cpu'):

        images, labels, meta = sequence
        seq = Sequence.from_meta(meta)[0]
        num_objects = len(seq.obj_ids)

        first_image = [images[fid].to(self.device) for fid in seq.first_frames]
        first_labels = [labels[fid].to(self.device) for fid in seq.first_frames]

        # Initialize the model

        self.eval()
        self.initialize(first_image, first_labels, seq)

        # Iterate over frames

        outputs = []
        desc = f"{j+1}/{n_seqs}, {seq.name}, {num_objects} object{'s' if num_objects > 1 else ''}"
        for frame in tqdm(seq.frames, desc=desc, ncols=100):
            masks, s = self.track(images[frame].to(self.device))
            outputs.append(masks.detach().to(out_device))
        outputs = torch.stack(outputs, dim=0)

        # Pad to full length and save the result. Ground truth is inserted where known.

        for i, obj_id in enumerate(seq.obj_ids):
            f0 = seq.frames.index(seq.first_frames[i])
            init_lb = first_labels[i] == obj_id
            gt_frame = init_lb.squeeze(1).float()
            outputs[f0][i] = gt_frame

        segs, obj_ids_all = self.merge_segmentations(outputs.permute(1, 0, 2, 3), seq.obj_ids)
        out_labels = obj_ids_all[segs.argmax(dim=0)].unsqueeze(1)  # (num_frames, 1, H, W)

        return seq, out_labels

    def initialize(self, first_images, init_labels, seq):

        self.num_objects = len(first_images)
        self.targets = dict()

        # Augment first image and extract features

        for i, first_image in enumerate(first_images):

            obj_id = seq.obj_ids[i]
            f0 = seq.frames.index(seq.first_frames[i])

            disc = Discriminator(self.disc_params)
            first_mask = (init_labels[i] == obj_id).byte()
            target = TargetObject(obj_id=obj_id, index=i, discriminator=disc, start_frame=f0, start_mask=first_mask)

            im, msk = self.augment(first_image[0], first_mask[0])
            ft = self.feature_extractor.no_grad_forward(im.to(self.device), [disc.params.layer], chunk_size=1)
            disc.init(ft, msk.to(self.device))
            self.targets[obj_id] = target

        self.current_frame = 0

    def update(self, ys, sz):

        out = torch.zeros((self.num_objects, *sz)).to(self.device)

        for i, (obj_id, t1) in enumerate(self.targets.items()):

            if t1.start_frame < self.current_frame:
                for obj_id2, t2 in self.targets.items():
                    if obj_id != obj_id2 and t2.start_frame == self.current_frame:
                        ys[obj_id] *= (1 - t2.start_mask).float()

                out[t1.index] = ys[obj_id]

        fg = torch.clamp(out, 1e-7, 1 - 1e-7)
        bg = torch.min((1 - fg), dim=0, keepdim=True)[0]
        p = torch.cat((bg, fg), dim=0)
        segs = F.softmax(p / (1 - p), dim=0)
        inds = segs.argmax(dim=0)

        for i in range(out.shape[0]):
            mask = inds == (i+1)
            out[i, mask] = segs[i+1,mask]
            out[i, mask == 0] = 0.0

        for obj_id, target in self.targets.items():
            if target.start_frame < self.current_frame and self.disc_params.update_filters:
                target.discriminator.update(out[target.index].unsqueeze(0).unsqueeze(0))

    def track(self, image):

        sz = image.shape[-2:]
        features = self.feature_extractor(image)

        ys = dict()
        scores = dict()

        for i, (obj_id, target) in enumerate(self.targets.items()):

            if target.start_frame < self.current_frame:
                d = target.discriminator
                score_layers = [d.params.layer]
                x = {L: features[L] for L in score_layers}
                x = d.preprocess_sample(x)
                s = d.apply(x)

                scores[obj_id] = s

                s = [s[L] for L in score_layers]
                y = self.refiner(s, features, sz)
                y = interpolate(y, image.shape[-2:])
                y = torch.sigmoid(y)

                ys[obj_id] = y.detach()

        if not self.training and self.current_frame > 0:
            self.update(ys, image.shape[-2:])

        out = torch.zeros((self.num_objects, *sz))
        for i, (obj_id, target) in enumerate(self.targets.items()):
            if target.start_frame < self.current_frame:
                out[target.index] = interpolate(ys[obj_id], sz)

        self.current_frame += 1

        return out, scores
