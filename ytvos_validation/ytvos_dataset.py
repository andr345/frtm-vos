from pathlib import Path
from PIL import Image
import numpy as np
import json

import torch
from torch.utils.data import Dataset


class Sequence:
    """ Sequence specification: target initialization frames and video frames. """

    def __init__(self, name=None, obj_ids=None, first_frames=None, frames=None):
        """
        :param name:            Sequence name
        :param obj_ids:         Object(s) to track (list of ints)
        :param first_frames:    Frame names for for initial training on the object(s)
        :param frames:          List of sequence frame names
        """
        self.name = name
        self.obj_ids = obj_ids
        self.first_frames = first_frames
        self.frames = frames

    def __repr__(self):
        return "Sequence: " + str(vars(self))

    def encoded(self):
        """ Get an object representation that will pass through the DataLoader's collate function unchanged. """
        v = json.dumps(vars(self))
        return v

    @staticmethod
    def from_meta(meta):
        """ Build Sequence objects from meta data passed through Pytorch's collate function """
        seqs = [Sequence(**json.loads(m)) for m in meta]
        return seqs


class YouTubeVOSTestDataset(Dataset):
    """ The Youtube-VOS dataset for testing and validation on Codalab. """

    def __init__(self, dset_path, name="valid", year=2018, sequences=None, restart=None):
        """
        :param dset_path: Dataset root path
        :param parameters: Dataset configuration dict
        """
        super().__init__()

        print(f"Evaluating on YouTubeVOS {year}, {name}")
        dset_path = dset_path / ("%d/%s" % (year, name))

        self.dset_path = Path(dset_path)
        self.jpeg_path = self.dset_path / "JPEGImages"
        self.anno_path = self.dset_path / "Annotations"

        if name.endswith("_all_frames") and name != 'test_all_frames':
            # Get the meta data from the not-all-frames variant
            meta_dir = self.dset_path.name.split("_", 1)[0]  # Extract "<x>" from "<x>_all_frames"
            meta_path = self.dset_path.parent / meta_dir
        else:
            meta_path = self.dset_path

        meta_data = json.load(open(meta_path / 'meta.json'))
        self.ytvos = meta_data

        if sequences is None:
            self.selected_sequences = self.ytvos['videos'].keys()
        else:
            self.selected_sequences = set(sequences)
        self.first_frames = dict()

        if restart is not None:
            if restart not in self.selected_sequences:
                print(f"{restart} not found in {self.__class__.__name__}, will not restart from there")
            else:
                s = list(sorted(self.selected_sequences))
                self.selected_sequences = s[s.index(restart):]

        self.sequences = []

        for seq_name in self.ytvos['videos'].keys():
            if seq_name not in self.selected_sequences:
                continue

            # Return all objects ids in one label map
            obj_ids = list(sorted(self.ytvos['videos'][seq_name]['objects'].keys()))
            seq = self.ytvos['videos'][seq_name]
            if name.startswith("valid"):
                first_frames = [seq['objects'][obj_id]['frames'][0] for obj_id in obj_ids]
            elif name.startswith("test"):
                first_frames = [seq['objects'][obj_id][0] for obj_id in obj_ids]
            else:
                raise ValueError
            self.sequences.append(Sequence(seq_name, obj_ids=list(map(int, obj_ids)), first_frames=first_frames))
            self.first_frames[seq_name] = first_frames
        pass

    @staticmethod
    def imread(filename):
        x = np.array(Image.open(filename))
        x = np.atleast_3d(x).transpose(2, 0, 1)
        x = torch.from_numpy(np.ascontiguousarray(x))
        return x

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, item):

        seq = self.sequences[item]

        # Load all frames

        im_files = list(sorted((self.jpeg_path / seq.name).glob("*.jpg")))
        seq.frames = [f.stem for f in im_files]

        images, labels = dict(), dict()
        for fname in im_files:
            f = fname.stem
            images[f] = self.imread(fname)
            fname = self.anno_path / seq.name / (f + ".png")
            if fname.exists():
                lb = np.array(Image.open(fname))
                labels[f] = torch.from_numpy(lb).unsqueeze(0)

        return images, labels, seq.encoded()
