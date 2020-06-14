from pathlib import Path
import json
from collections import OrderedDict as odict
from easydict import EasyDict as edict
from PIL import Image
from tqdm import tqdm
import cv2
import numpy as np
import random

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SampleSpec:

    def __init__(self, seq_name=None, obj_id=None, frames=None, frame0_id=None):
        self.seq_name = seq_name
        self.obj_id = obj_id
        self.frames = frames
        self.frame0_id = frame0_id

    def __repr__(self):
        return "SampleSpec: " + str(vars(self))

    def encoded(self):
        v = json.dumps(vars(self))
        return v

    @staticmethod
    def from_encoded(meta):
        specs = [SampleSpec(**json.loads(m)) for m in meta]
        return specs


class TrainingDataset(Dataset):

    def __init__(self, name, dset_path):
        super().__init__()
        self.dset_path = Path(dset_path)
        self.name = name

    def load_meta(self):

        meta_file = Path(__file__).resolve().parent / (self.name + "_meta.pth")
        if meta_file.exists():
            return torch.load(meta_file)

        print("Caching occlusions for %s, please wait." % self.anno_path)

        frame_names = dict()
        label_pixel_counts = dict()

        paths = [self.anno_path / seq for seq in sorted(self.sequences)]
        for k, p in enumerate(tqdm(paths)):

            frames = []
            num_objects = 0

            # Gather per-frame stats

            seq_lb_files = list(sorted(p.glob("*.png")))
            for lb_path in seq_lb_files:

                lb = np.array(Image.open(lb_path))
                obj_ids, counts = np.unique(lb, return_counts=True)
                frames.append((obj_ids, counts))
                num_objects = max(num_objects, max(obj_ids))

            # Populate a matrix of object pixel counts

            px_counts = np.zeros((len(frames), num_objects + 1))

            for i, (obj_ids, counts) in enumerate(frames):
                for oid, cnt in zip(obj_ids, counts):
                    px_counts[i, oid] = cnt

            frame_names[p.stem] = [f.stem for f in seq_lb_files]
            label_pixel_counts[p.stem] = (px_counts, np.max(px_counts, axis=0))

        # Generate object occlusions information and save

        occlusions = self._generate_occlusions(label_pixel_counts)
        meta = dict(frame_names=frame_names, occlusions=occlusions)
        torch.save(meta, meta_file)

        return meta

    def generate_samples(self, epoch_samples, epoch_repeats, min_seq_length, sample_size):

        d = self.load_meta()
        self.occlusions = d['occlusions']
        self.frame_names = d['frame_names']

        sequences = []
        for seq_name in self.sequences:
            if self.sequence_length(seq_name) < min_seq_length:
                continue
            for obj_id in self.object_ids(seq_name)[1:].tolist():
                sequences.append(edict(name=seq_name, obj_id=obj_id))

        if epoch_samples > 0:
            sequences = random.sample(sequences, epoch_samples)

        self.specs = []
        for seq in sequences:
            for rep in range(epoch_repeats):
                spec = self.sample_random_image_set(seq.name, obj_id=seq.obj_id, size=sample_size)
                self.specs.append(spec)

    def sample_random_image_set(self, seq_name, obj_id, size=3):
        """  Create a SampleSpec object representing a (random) set of frames from a sequence.
        :param seq_name:       Sequence name
        :param obj_id:        Object to track (int)
        :param size:           Set size > 1
        :return: SampleSpec object
        """
        object_visible = self.object_visibility(seq_name, [obj_id], merge_objects=True)

        possible_frames = np.where(object_visible)[0]
        frames = np.random.choice(possible_frames, size=1, replace=False).tolist()
        first_frame = frames[0]

        num_frames = self.sequence_length(seq_name)
        allframes = np.arange(num_frames)
        allframes = allframes[allframes != first_frame]
        frames = np.random.choice(allframes, size=size, replace=False).tolist()

        return SampleSpec(seq_name, obj_id, frames=[first_frame, *frames[1:]], frame0_id=first_frame)

    def object_ids(self, seq_name):
        """ Find the ids of objects seen in the sequence. id 0 == background """
        assert self.occlusions is not None
        occlusions = self.occlusions[seq_name]
        always_occluded = occlusions.sum(axis=0) == occlusions.shape[0]
        object_ids = np.where(np.invert(always_occluded))[0]

        return object_ids

    def object_visibility(self, seq_name, obj_ids, merge_objects=False):
        """ Get boolean vector of per-frame object visibility in the named sequence.
        :param seq_name:  Sequence name
        :param obj_ids:  Zero (None), one (int) or more (list) object ids.
                         If zero, all objects (except the background) are selected
        :param merge_objects:   If true, the visibilities of multiple objects are merged.
        :return:
        """
        assert self.occlusions is not None

        visible = np.invert(self.occlusions[seq_name])

        if obj_ids is None:
            visible = visible[:, 1:]
        else:
            visible = visible[:, obj_ids]

        if visible.ndim == 1:
            visible = np.expand_dims(visible, axis=1)

        if merge_objects:
            visible = visible.any(axis=1)

        if visible.ndim == 1:
            visible = np.expand_dims(visible, axis=1)

        return visible

    def sequence_length(self, seq_name):
        return self.occlusions[seq_name].shape[0]

    def __len__(self):
        return len(self.specs)

    def __getitem__(self, item):

        spec = self.specs[item]
        images = []
        labels = []

        frame_names = self.frame_names[spec.seq_name]
        for f in spec.frames:
            frame = frame_names[f]

            im = np.array(Image.open(self.jpeg_path / spec.seq_name / (frame + ".jpg")))
            s = 480 / im.shape[0]
            im = cv2.resize(im, (854, 480), cv2.INTER_AREA if (s < 1.0) or (self.name == 'davis') else cv2.INTER_CUBIC)
            im = torch.from_numpy(im.transpose(2, 0, 1))
            images.append(im)

            lb = np.array(Image.open(self.anno_path / spec.seq_name / (frame + ".png")))
            lb = (lb == spec.obj_id).astype(np.uint8)  # Relabel selected object to id 1
            lb = torch.as_tensor(lb, dtype=torch.float32).view(1, 1, *lb.shape[:2])
            lb = F.interpolate(lb, (480, 854), mode='nearest').byte().squeeze(0)
            labels.append(lb)

        return images, labels, spec.encoded()


class DAVISDataset(TrainingDataset):

    def __init__(self, dset_path: Path, epoch_repeats=1, epoch_samples=0, min_seq_length=4, sample_size=3):
        super().__init__("davis", dset_path)

        self.jpeg_path = self.dset_path / "JPEGImages" / "480p"
        self.anno_path = self.dset_path / "Annotations" / "480p"
        self.sequences = [s.strip() for s in open(self.dset_path / "ImageSets/2017/train.txt").readlines()]

        self.generate_samples(epoch_samples, epoch_repeats, min_seq_length, sample_size)

    def _generate_occlusions(self, label_pixel_counts):
        """ Generate per-frame, per-object occlusion flags
            Each sequence is an (N, M) boolean array of N frames and M object ids. True/False if occluded/visible.
            object 0 is the background. """

        occlusions = odict()

        min_px = 100  # Hard minimum

        never_occluded = ['bus', 'car-turn', 'drift-turn', 'kid-football', 'koala',
                          'mallard-fly', 'motocross-bumps', 'motorbike',
                          'rallye', 'snowboard', 'train', 'upside-down']

        for seq_name in tqdm(self.sequences):

            px_counts, max_counts = label_pixel_counts[seq_name]
            seq_length = len(list((self.jpeg_path / seq_name).glob("*.jpg")))

            if seq_name in never_occluded:
                occ = np.zeros(shape=px_counts.shape, dtype=np.bool)
            else:

                # Pixel fraction

                if seq_name in ('bmx-bumps', 'disk-jockey'):
                    occ_threshold = 0.5
                elif seq_name in ('boxing-fisheye', 'cat-girl', 'dog-gooses'):
                    occ_threshold = 0.2
                elif seq_name in ('tractor-sand', 'drone'):
                    occ_threshold = 0.1
                else:
                    occ_threshold = 0.25

                occ = (px_counts / (max_counts + 0.001)) < occ_threshold
                occ = occ + (max_counts == 0)

            # Sequence specific tweaks

            if seq_name == 'classic-car':
                occ[:56, :] = False
            elif seq_name == 'drone':
                occ[:17, 1] = False  # Red quad
                occ[24:60, 1] = False
            elif seq_name == 'night-race':
                occ[:29, :] = False
                occ[:, 2] = False  # Green car

            occ = occ + (px_counts < min_px)  # Apply a hard minimum

            occlusions[seq_name] = occ

        return occlusions


class YouTubeVOSDataset(TrainingDataset):

    def __init__(self, dset_path, epoch_samples=4000, epoch_repeats=1, min_seq_length=4, sample_size=3, year=2018):
        super().__init__("ytvos" + str(year), dset_path)

        self.jpeg_path = self.dset_path / "train" / "JPEGImages"
        self.anno_path = self.dset_path / "train" / "Annotations"
        self.sequences = [s.strip() for s in open(Path(__file__).resolve().parent / "ytvos_jjtrain.txt").readlines()]

        self.generate_samples(epoch_samples, epoch_repeats, min_seq_length, sample_size)

    def _generate_occlusions(self, label_pixel_counts):
        """ Generate per-frame, per-object occlusion flags
            Each sequence is an (N, M) boolean array of N frames and M object ids. True/False if occluded/visible.
            object 0 is the background. """

        occlusions = odict()
        for seq_name, (px_counts, max_counts) in label_pixel_counts.items():
            occlusions[seq_name] = (px_counts < 100)

        return occlusions
