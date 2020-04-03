from collections import OrderedDict as odict
import numpy as np

from lib.image import imread
from lib.utils import text_bargraph
import lib.davis as utils


def evaluate_dataset(dset, results_path, measure='J', to_file=True):

    results = odict()
    dset_scores = []
    dset_decay = []
    dset_recall = []

    if to_file:
        f = open(results_path / ("evaluation-%s.txt" % measure), "w")

    def _print(msg):
        print(msg)
        if to_file:
            print(msg, file=f)
            f.flush()

    for j, sequence in enumerate(dset):

        # Load all frames

        annotations = odict()
        segmentations = odict()
        for file in sequence.annos:
            lb = imread(file)
            annotations[file.stem] = (lb != 0).byte() if sequence.merge_objects else lb
            segmentations[file.stem] = imread(results_path / sequence.name / file.name)

        # Find object ids and starting frames

        object_info = dict()
        for obj_id in sequence.obj_ids:
            for frame, obj_ids in sequence.start_frames.items():
                if obj_id in obj_ids:
                    assert obj_id not in object_info  # Only one start frame per object
                    object_info[obj_id] = frame
        assert 0 not in object_info  # Because a background object would be weird

        # Evaluate

        n_seqs = len(dset)
        n_objs = len(object_info)
        seq_name = sequence.name

        _print("%d/%d: %s: %d object%s" % (j + 1, n_seqs, seq_name, n_objs, "s" if n_objs > 1 else ""))
        r = utils.evaluate_sequence(segmentations, annotations, object_info, measure=measure)
        results[seq_name] = r

        # Print scores, per frame and object, ignoring NaNs

        per_obj_score = []  # Per-object accuracies, averaged over the sequence
        per_frame_score = []  # Per-frame accuracies, averaged over the objects

        for obj_id, score in r['raw'].items():
            per_frame_score.append(score)
            s = utils.mean(score)  # Sequence average for one object
            per_obj_score.append(s)
            if n_objs > 1:
                _print("joint {obj}: acc {score:.3f} ┊{apf}┊".format(obj=obj_id, score=s, apf=text_bargraph(score)))

        # Print mean object score per frame and final score

        dset_decay.extend(r['decay'])
        dset_recall.extend(r['recall'])
        dset_scores.extend(per_obj_score)

        seq_score = utils.mean(per_obj_score)  # Final score
        seq_mean_score = utils.nanmean(np.array(per_frame_score), axis=0)  # Mean object score per frame

        # Print sequence results

        _print("final  : acc {seq:.3f} ({dset:.3f}) ┊{apf}┊".format(
            seq=seq_score, dset=np.mean(dset_scores), apf=text_bargraph(seq_mean_score)))

    _print("%s: %.3f, recall: %.3f, decay: %.3f" % (measure, utils.mean(dset_scores), utils.mean(dset_recall),
                                                    utils.mean(dset_decay)))
    if to_file:
        f.close()
