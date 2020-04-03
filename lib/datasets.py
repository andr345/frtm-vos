from pathlib import Path
import json
import torch
from torch.utils.data import Dataset
from collections import defaultdict
from lib.image import imread


def transpose_dict(d):
    dt = defaultdict(list)
    for k, v in d.items():
        dt[v].append(k)
    return dt


class FileSequence(Dataset):
    """ Inference-only dataset. A sequence backed by jpeg images and start label pngs. """

    def __init__(self, dset_name, seq_name, jpeg_path: Path, anno_path: Path, start_frames: dict, merge_objects=False,
                 all_annotations=False):

        self.dset_name = dset_name
        self.name = seq_name

        self.images = list(sorted(jpeg_path.glob("*.jpg")))
        self.preloaded_images = None
        self.anno_path = anno_path
        self.start_frames = dict(transpose_dict(start_frames))  # key: frame, value: list of object ids
        self.obj_ids = list(start_frames.keys()) if not merge_objects else [1]
        self.frame_names = [f.stem for f in self.images]
        self.merge_objects = merge_objects
        if all_annotations:
            self.annos = list(sorted(anno_path.glob("*.png")))

    def __len__(self):
        return len(self.images)

    def __getitem__(self, item):

        if self.preloaded_images is not None:
            im = self.preloaded_images[item]
        else:
            im = imread(self.images[item])
        lb = []
        f = self.frame_name(item)
        obj_ids = self.start_frames.get(f, [])

        if len(obj_ids) > 0:
            lb = imread(self.anno_path / (f + ".png"))
            if self.merge_objects:
                lb = (lb != 0).byte()
                obj_ids = [1]
            else:
                # Suppress labels of objects not in their start frame (primarily for YouTubeVOS)
                suppressed_obj_ids = list(set(lb.unique().tolist()) - set([0] + obj_ids))
                for obj_id in suppressed_obj_ids:
                    lb[lb == obj_id] = 0

        return im, lb, obj_ids

    def frame_name(self, item):
        return self.images[item].stem

    def preload(self, device):
        """ Preload all images and upload them to the GPU. """
        self.preloaded_images = [imread(f).to(device) for f in self.images]

    def __repr__(self):
        return "%s: %s, %d frames" % (self.dset_name, self.name, len(self.images))


class DAVISDataset:

    def __init__(self, path, year: str, split: str, restart: str=None, sequences: (list, tuple)=None,
                 all_annotations=False):

        self.dset_path = Path(path).expanduser().resolve()
        if not self.dset_path.exists():
            print("Dataset directory '%s' not found." % path)
            quit(1)

        self.jpeg_path = self.dset_path / "JPEGImages" / "480p"
        self.anno_path = self.dset_path / "Annotations" / "480p"
        imset = self.dset_path / "ImageSets" / year / (split + ".txt")
        self.sequences = [s.strip() for s in sorted(open(imset).readlines())]
        self.name = "dv%s%s" % (year, split)
        self.year = year
        self.all_annotations = all_annotations

        if sequences is not None:
            assert set(sequences).issubset(self.sequences)
            self.sequences = list(sorted(set(self.sequences).intersection(sequences)))
        if restart is not None:
            assert restart in self.sequences
            self.sequences = self.sequences[self.sequences.index(restart):]

        self.start_frames = dict()
        for seq in self.sequences:
            f0 = "00000"  # In DAVIS, all objects appear in the first frame
            obj_ids = torch.unique(imread(self.anno_path / seq / (f0 + ".png"))).tolist()
            self.start_frames[seq] = {obj_id: f0 for obj_id in sorted(obj_ids) if obj_id != 0}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        seq = self.sequences[item]
        return FileSequence(self.name, seq, self.jpeg_path / seq, self.anno_path / seq, self.start_frames[seq],
                            merge_objects=self.year == '2016', all_annotations=self.all_annotations)


class YouTubeVOSDataset:

    def __init__(self, path, year: str, split: str, restart: str=None, sequences: (list, tuple)=None,
                 all_annotations=False):

        self.dset_path = Path(path).expanduser().resolve()
        if not self.dset_path.exists():
            print("Dataset directory '%s' not found." % path)
            quit(1)

        self.name = "ytvos%s%s" % (year, split)
        self.year = year
        self.all_annotations = all_annotations

        if split in ('train', 'train_all_frames', 'jjval', 'jjval_all_frames'):
            im_split = "train_all_frames" if split.endswith("_all_frames") else "train"
            self.jpeg_path = self.dset_path / im_split / "JPEGImages"
            self.anno_path = self.dset_path / "train" / "Annotations"
            imset = Path(__file__).parent / ("ytvos_jjvalid.txt" if split.startswith('jjval') else "ytvos_jjtrain.txt")
            self.sequences = [s.strip() for s in sorted(open(imset).readlines())]
            self.meta = json.load(open(self.dset_path / "train" / "meta.json"))['videos']
        elif split in ('test', 'test_all_frames', 'valid', 'valid_all_frames'):
            im_split = split
            split = split[:-len('_all_frames')] if split.endswith('_all_frames') else split
            self.jpeg_path = self.dset_path / im_split / "JPEGImages"
            self.anno_path = self.dset_path / split / "Annotations"
            self.sequences = [s.name for s in sorted(self.anno_path.glob("*")) if s.is_dir()]
            self.meta = json.load(open(self.dset_path / split / "meta.json"))['videos']

        if sequences is not None:
            assert set(sequences).issubset(self.sequences)
            self.sequences = list(sorted(set(self.sequences).intersection(sequences)))
        if restart is not None:
            assert restart in self.sequences
            self.sequences = self.sequences[self.sequences.index(restart):]

        self.start_frames = dict()
        for seq in self.sequences:
            self.start_frames[seq] = {int(obj_id): v['frames'][0] for obj_id, v in self.meta[seq]['objects'].items()}

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):
        seq = self.sequences[item]
        return FileSequence(self.name, seq, self.jpeg_path / seq, self.anno_path / seq, self.start_frames[seq],
                            all_annotations=self.all_annotations)
