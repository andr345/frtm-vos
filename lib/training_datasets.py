import json
from pathlib import Path
from collections import OrderedDict as odict
from collections import defaultdict as ddict
from easydict import EasyDict as edict
from PIL import Image
import numpy as np
import cv2
import random
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset


class SampleSpec:

    def __init__(self, seq_name=None, obj_id=None, frames=None, frame0_id=None):

        self.seq_name = seq_name
        self.obj_id = int(obj_id)
        self.frames = frames
        self.frame0_id = frame0_id  # For caching target models

    def __repr__(self):
        return "SampleSpec: " + str(vars(self))

    def encoded(self):
        v = json.dumps(vars(self))
        return v

    @staticmethod
    def from_encoded(meta):
        specs = [SampleSpec(**json.loads(m)) for m in meta]
        return specs


class DatasetMeta:

    @staticmethod
    def count_label_pixels(anno_path, sequences, resize_to=None):
        """ Count the label pixels per object, per frame. """
        label_pixel_counts = dict()

        for k, seq in enumerate(tqdm(sequences)):

            seq_px_counts = ddict(dict)
            max_px_counts = ddict(int)

            for lb_path in sorted((anno_path / seq).glob("*.png")):

                lb = np.array(Image.open(lb_path))

                # Resize to the shape the trainer will use
                if resize_to is not None:
                    lb = cv2.resize(lb, dsize=resize_to, interpolation=cv2.INTER_NEAREST)

                # Count pixels per object
                object_ids, counts = np.unique(lb, return_counts=True)
                for obj_id, n in zip(object_ids.tolist(), counts.tolist()):
                    seq_px_counts[obj_id][lb_path.stem] = n
                    max_px_counts[obj_id] = max(n, max_px_counts[obj_id])

            for obj_id, max_n in max_px_counts.items():
                seq_px_counts[obj_id]['max'] = max_px_counts[obj_id]

            label_pixel_counts[seq] = dict(seq_px_counts)

        return label_pixel_counts

    def get_label_pixels_counts(self, lpc_file, sequences, resize_to=None):

        if not lpc_file.exists():
            print("Caching label pixel counts in %s, please wait." % self.anno_path)
            label_pixel_counts = self.count_label_pixels(self.anno_path, sequences, resize_to=resize_to)
            json.dump(label_pixel_counts, open(lpc_file, "w"))

        label_pixel_counts = json.load(open(lpc_file))
        return label_pixel_counts

    def get_all_frames(self, sequences):

        all_frames = dict()
        for seq in sequences:
            seq_frames = []
            for ff in self.label_pixel_counts[seq].values():
                seq_frames.extend(ff)
            seq_frames = set(seq_frames)
            seq_frames.discard('max')
            all_frames[seq] = list(sorted(seq_frames))

        return all_frames

    def object_ids(self, seq_name):
        """ Find the ids of objects seen in the sequence. id 0 == background """
        return list(self.visibilities[seq_name].keys())

    def object_visibility(self, seq_name, obj_id):
        return self.visibilities[seq_name][obj_id]

    def sequence_frames(self, seq_name):
        return self.all_frames[seq_name]

    def sequence_length(self, seq_name):
        return len(self.all_frames[seq_name])

    def sample_random_image_set(self, seq_name, obj_id, size=3):
        """  Select random frames from a sequence and create a sample
        :param seq_name:       Sequence name / identifier
        :param obj_id:         Object to track (int)
        :param size:           Number of frames to select > 1
        :return: SampleSpec object
        """
        visible_frames = self.object_visibility(seq_name, obj_id)
        first_frame = np.random.choice(visible_frames, size=1).tolist()[0]

        allframes = set(self.sequence_frames(seq_name))
        allframes.discard(first_frame)
        frames = np.random.choice(list(sorted(allframes)), size=size - 1, replace=False).tolist()
        frame0_id = self.sequence_frames(seq_name).index(first_frame)

        return SampleSpec(seq_name=seq_name, obj_id=obj_id, frames=[first_frame, *frames], frame0_id=frame0_id)

    def _generate_samples(self, sequences, sample_size, epoch_samples, epoch_repeats):

        samples = []

        log_seqs = []
        for seq in sequences:
            # Create logical sequences from one video, each with different target objects
            for obj_id in self.object_ids(seq):
                log_seqs.append(edict(name=seq, obj_id=obj_id))

        # Select from the logical sequences
        for _ in range(epoch_repeats):
            epoch_seqs = random.sample(log_seqs, epoch_samples) if epoch_samples > 0 else log_seqs
            # Selecting random frames from each sequence and build the sample
            for seq in epoch_seqs:
                s = self.sample_random_image_set(seq.name, obj_id=seq.obj_id, size=sample_size)
                if s is not None:
                    samples.append(s)

        return samples

    def load_sample(self, sample: SampleSpec, resize_to=(480, 854)):

        images = []
        labels = []

        for f in sample.frames:
            im_path = str(self.jpeg_path / sample.seq_name / (f + ".jpg"))
            lb_path = str(self.anno_path / sample.seq_name / (f + ".png"))

            im = torch.from_numpy(np.array(Image.open(im_path)).transpose(2, 0, 1))
            lb = torch.from_numpy(np.array(Image.open(lb_path)))
            lb = lb.view(1, *lb.shape[-2:])
            lb = (lb == sample.obj_id).byte()  # Select and rename the target object

            if resize_to is not None:
                im = F.interpolate(im.unsqueeze(0).float(), resize_to, mode='bicubic', align_corners=False).byte().squeeze(0)
                lb = F.interpolate(lb.unsqueeze(0).float(), resize_to, mode='nearest').byte().squeeze(0)

            images.append(im)
            labels.append(lb)

        meta = sample.encoded()
        return images, labels, meta


class DAVISDataset(Dataset, DatasetMeta):

    def _get_visibilities(self, vis_file):

        if vis_file.exists():
            return json.load(open(vis_file))

        min_px = 100

        never_occluded = ['bus', 'car-turn', 'drift-chicane', 'drift-straight', 'drift-turn', 'kid-football', 'koala',
                          'mallard-fly', 'mbike-trick', 'motocross-bumps', 'motocross-jump', 'motorbike', 'paragliding-launch',
                          'parkour', 'rallye', 'scooter-black', 'snowboard', 'soapbox', 'train',
                          'upside-down']

        visibilities = odict()

        for seq_name, seq_objects in self.label_pixel_counts.items():

            seq_visibilities = ddict(list)

            if seq_name in ('bmx-bumps', 'disk-jockey', 'lab-coat'):
                occ_threshold = 0.5
            elif seq_name in ('boxing-fisheye', 'cat-girl', 'dog-gooses'):
                occ_threshold = 0.2
            elif seq_name in ('tractor-sand', 'dogs-jump', 'drone'):
                occ_threshold = 0.1
            else:
                occ_threshold = 0.25

            for obj_id, frames in seq_objects.items():
                if obj_id == '0':
                    continue

                if seq_name in never_occluded:
                    frames = [f for f in frames if f != 'max']
                    seq_visibilities[obj_id] = frames
                    continue

                max_count = frames['max']
                for frame, px_count in frames.items():
                    if frame == 'max':
                        continue
                    f = int(frame)

                    # Occlude based on how small the object is
                    occ = (px_count / (max_count + 0.001)) < occ_threshold
                    occ = occ or (max_count == 0)

                    # Manual tweaks
                    if seq_name == 'classic-car':
                        if f < 56:
                            occ = False
                    elif seq_name == 'dogs-jump' and obj_id == '2':
                        occ = False  # Green dog
                    elif seq_name == 'drone' and obj_id == '1':
                        if f < 17 or (24 <= f < 60):
                            occ = False  # Red quad
                    elif seq_name == 'lab-coat' and int(obj_id) >= 3:
                        occ[:, 3:] = False
                    elif seq_name == 'night-race':
                        if f < 29:
                            occ = False
                        if obj_id == '2':
                            occ = False  # Green car

                    occluded = occ or (px_count < min_px)

                    if not occluded:
                        seq_visibilities[obj_id].append(frame)

            visibilities[seq_name] = dict(seq_visibilities)
        json.dump(visibilities, open(vis_file, "w"))

        return json.load(open(vis_file))

    def __init__(self, dset_path, sample_size=3, epoch_samples=0, epoch_repeats=1):

        self.dset_path = Path(dset_path)
        self.jpeg_path = self.dset_path / "JPEGImages" / "480p"
        self.anno_path = self.dset_path / "Annotations" / "480p"

        self.sample_size = sample_size
        self.epoch_samples = epoch_samples
        self.epoch_repeats = epoch_repeats

        meta_path = Path(__file__).resolve().parent / "dset_meta"
        assert meta_path.exists()

        split = self.dset_path / "ImageSets/2017/train.txt"
        lpc_file = meta_path / "dv2017_label_pixel_counts.json"
        vis_file = meta_path / "dv2017_visibilities.json"

        split = [s.strip() for s in sorted(open(split).readlines())]
        self.label_pixel_counts = self.get_label_pixels_counts(lpc_file, split, resize_to=(854, 480))
        self.visibilities = self._get_visibilities(vis_file)
        self.all_frames = self.get_all_frames(split)

        self.sequences = split
        self.samples = self._generate_samples(self.sequences, self.sample_size, self.epoch_samples, self.epoch_repeats)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        images, labels, meta = self.load_sample(self.samples[item], resize_to=(480, 854))
        return images, labels, meta


class YouTubeVOSDataset(Dataset, DatasetMeta):

    def _get_visibilities(self, vis_file):

        min_px = 100

        if not vis_file.exists():
            visibilities = odict()
            for seq_name, seq_objects in self.label_pixel_counts.items():
                seq_visibilities = ddict(list)
                for obj_id, frames in seq_objects.items():
                    if obj_id == '0':
                        continue
                    for f, n in frames.items():
                        if f == 'max':
                            continue
                        if n > min_px:
                            seq_visibilities[obj_id].append(f)
                visibilities[seq_name] = dict(seq_visibilities)
            json.dump(visibilities, open(vis_file, "w"))

        return json.load(open(vis_file, "r"))

    def _get_clean_split(self, sequences, clean_split_file):

        if not clean_split_file.exists():

            # Remove YouTubeVOS training sequences with unusual aspect ratios

            removed = 0
            good_ar_split = []

            for k, seq in enumerate(tqdm(sequences)):
                src_annos = list(sorted((self.anno_path / seq).glob("*.png")))

                w, h = Image.open(src_annos[0]).size
                s = 480 / h
                w = int(s * w + 0.5)

                # Remove very wide or very narrow sequences
                if w < 840 or w > 860:
                    removed += 1
                    continue
                good_ar_split.append(seq)

            # Remove sequences where objects appear in just a small number of frames.
            # Whenever this happens, it indicates that object ids mutate.

            required_frames = 4
            sequences = []

            for seq in good_ar_split:
                bad_seq = False
                for obj_id, frames in self.visibilities[seq].items():
                    bad_seq = bad_seq or len(frames) < required_frames
                if not bad_seq:
                    sequences.append(seq)

            with open(str(clean_split_file), "w") as f:
                for s in sequences:
                    print(s, file=f)

        sequences = [s.strip() for s in sorted(open(clean_split_file).readlines())]
        return sequences

    def __init__(self, dset_path, sample_size=3, min_seq_length=3, epoch_samples=0, epoch_repeats=1):

        self.dset_path = Path(dset_path)
        self.jpeg_path = self.dset_path / "train" / "JPEGImages"
        self.anno_path = self.dset_path / "train" / "Annotations"

        self.sample_size = sample_size
        self.epoch_samples = epoch_samples
        self.epoch_repeats = epoch_repeats

        meta_path = Path(__file__).resolve().parent / "dset_meta"
        assert meta_path.exists()

        jjval_split = meta_path / "ytvos_jjtrain.txt"
        clean_split = meta_path / "ytvos_clean_jjtrain.txt"
        lpc_file = meta_path / "ytvos2018_label_pixel_counts.json"
        vis_file = meta_path / "ytvos2018_visibilities.json"

        jjval_split = [s.strip() for s in sorted(open(jjval_split).readlines())]
        self.label_pixel_counts = self.get_label_pixels_counts(lpc_file, jjval_split, resize_to=(854, 480))
        self.visibilities = self._get_visibilities(vis_file)
        self.all_frames = self.get_all_frames(jjval_split)
        split = self._get_clean_split(jjval_split, clean_split)

        self.sequences = [seq for seq in split if self.sequence_length(seq) >= min_seq_length]
        self.samples = self._generate_samples(self.sequences, self.sample_size, self.epoch_samples, self.epoch_repeats)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        images, labels, meta = self.load_sample(self.samples[item], resize_to=(480, 854))
        return images, labels, meta
