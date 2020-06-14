from collections import defaultdict as ddict
from time import time

import torch
from torch.utils.data import ConcatDataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from lib.training_datasets import *
from lib.utils import AverageMeter


class Trainer:

    def __init__(self, name, model, optimizer, scheduler, dataset, checkpoints_path, log_path,
                 max_epochs, batch_size, num_workers=0, load_latest=True, save_interval=5,
                 stats_to_print=('stats/loss', 'stats/accuracy', 'stats/lr', 'stats/fcache_hits')):

        self.name = name
        self.model = model
        self.model = model
        self.optimizer = optimizer
        self.scheduler = scheduler

        self.dataset = dataset
        self.checkpoints_path = checkpoints_path / name
        self.checkpoints_path.mkdir(exist_ok=True, parents=True)
        self.log_path = log_path / name
        self.log = None

        self.epoch = 0
        self.max_epochs = max_epochs
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.save_interval = save_interval

        self.stats_to_print = stats_to_print

        self.stats = ddict(AverageMeter)

        if load_latest:
            checkpoints = list(sorted(self.checkpoints_path.glob("%s_ep*.pth" % name)))
            if len(checkpoints) > 0:
                self.load_checkpoint(checkpoints[-1])

    def load_checkpoint(self, file):

        print("Loading checkpoint", file)
        ckpt = torch.load(file, map_location='cpu')
        # assert ckpt['name'] == self.name
        self.epoch = ckpt['epoch']
        print("Starting epoch", self.epoch + 1)
        self.stats = ckpt['stats']
        self.model.load_state_dict(ckpt['model'])
        self.optimizer.load_state_dict(ckpt['optimizer'])
        self.scheduler.load_state_dict(ckpt['scheduler'])

    def save_checkpoint(self):

        ckpt = dict(name=self.name,
                    epoch=self.epoch,
                    stats=self.stats,
                    model=self.model.state_dict(),
                    optimizer=self.optimizer.state_dict(),
                    scheduler=self.scheduler.state_dict())
        torch.save(ckpt, self.checkpoints_path / ("%s_ep%04d.pth" % (self.name, self.epoch)))

    def upload_batch(self, batch):

        def recurse_scan(v):
            if torch.is_tensor(v):
                v = v.to(self.model.device)
            elif isinstance(v, (list, tuple)):
                v = [recurse_scan(x) for x in v]
            elif isinstance(v, dict):
                v = {k: recurse_scan(x) for k, x in v.items()}
            return v

        return recurse_scan(batch)

    def update_stats(self, new_stats, iteration, iters_per_epoch, runtime, do_print=False):

        for k, v in new_stats.items():
            self.stats[k].update(v)

        if not do_print:
            return

        header = "{self.epoch}: {iteration}/{iters_per_epoch}, sps={sps:.2f} ({sps_avg:.2f}), ".format(
            self=self, iteration=iteration, iters_per_epoch=iters_per_epoch,
            sps=self.batch_size / runtime.val, sps_avg=self.batch_size / runtime.avg)
        # sps: samples per second

        stats = []
        dec = 5
        for k, v in self.stats.items():
            if k in self.stats_to_print:
                k = k[6:] if k.startswith("stats/") else k
                s = '{k}={v.val:.{dec}f} ({v.avg:.{dec}f})'.format(k=k, v=v, dec=dec)
                stats.append(s)

        print(header + ", ".join(stats))

    def log_stats(self):

        if self.log is None:
            self.log = SummaryWriter(str(self.log_path))

        for k, v in self.stats.items():
            self.log.add_scalar(k, v.avg, self.epoch)

    def train(self):

        for epoch in range(self.epoch + 1, self.max_epochs + 1):

            self.epoch = epoch
            self.stats = ddict(AverageMeter)

            dset = ConcatDataset([eval(cls)(**params) for cls, params in self.dataset])

            loader = DataLoader(dset, batch_size=self.batch_size, num_workers=self.num_workers,
                                pin_memory=True, shuffle=True)
            t0 = None
            runtime = AverageMeter()

            for i, batch in enumerate(loader, 1):
                t0 = time() if t0 is None else t0  # Ignore loader startup pause

                self.optimizer.zero_grad()
                stats = self.model(*batch)
                self.optimizer.step()

                runtime.update(time() - t0)
                t0 = time()

                stats['stats/lr'] = self.scheduler.get_last_lr()[0]
                self.update_stats(stats, i, len(loader), runtime, do_print=True)

            self.scheduler.step()

            if self.epoch % self.save_interval == 0:
                self.save_checkpoint()

            self.log_stats()

        print("%s done" % self.name)
