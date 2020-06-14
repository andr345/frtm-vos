import torch
import torch.nn as nn
import torch.nn.functional as F

from lib.tensorlist import TensorList
from lib.utils import interpolate, conv
from .optimizer import MinimizationProblem, GaussNewtonCG
from .memory import Memory


class DiscriminatorLoss(MinimizationProblem):

    def __init__(self, x, y, filter_regs, precond, sample_weights,
                 net, pixel_weighting, compute_norm=False):
        super().__init__()

        self.training_samples = x
        self.training_labels = y
        self.y_size = y.shape[-2:]

        self.x = x
        self.y = y
        self.w = pixel_weighting

        self.filter_regs = TensorList(filter_regs)
        if sample_weights.size()[0] == 1:
            self.sample_weights = x.new_zeros(x.shape[0])
            self.sample_weights.fill_(sample_weights[0])
        else:
            self.sample_weights = sample_weights

        self.diag_M = TensorList(precond)
        self.pixel_weighting = pixel_weighting

        self.net = net
        self.compute_norm = compute_norm

    def initialize(self):

        a = self.sample_weights > 0.0
        self.x = self.training_samples[a]
        self.y = self.training_labels[a]
        self.w = self.pixel_weighting[a] * self.sample_weights[a].sqrt().view(-1, 1, 1, 1)

    def __call__(self, parameters: TensorList):

        s = self.net(self.x)
        s = F.interpolate(s, self.y_size, mode='bilinear', align_corners=False)
        residuals = self.w * (s - self.y)
        return TensorList([residuals, *(self.filter_regs * parameters)])

    def ip(self, a: TensorList, b: TensorList):
        return a.view(-1) @ b.view(-1)

    def ip_input(self, a: TensorList, b: TensorList):
        out = self.ip(a, b)
        sum_out = sum(o.unsqueeze(dim=0) for o in out)
        rep_sum = TensorList([])
        for o in out:
            rep_sum.extend(sum_out.clone())
        return rep_sum

    def M1(self, x: TensorList):
        return x / self.diag_M


class Discriminator(nn.Module):

    def forward(self, x):
        x = self.project(x)
        x = self.filter(x)
        return x

    def __init__(self, in_channels=1024, c_channels=96, out_channels=1,
                 init_iters=(5, 10, 10, 10, 10), update_iters=(10,), update_filters=True,
                 filter_reg=(1e-4, 1e-2), precond=(1e-4, 1e-2), precond_lr=0.1, CG_forgetting_rate = 75,
                 memory_size=80, train_skipping=8, learning_rate=0.1,
                 pixel_weighting=None, device=None, layer=None):
        super().__init__()

        self.project = conv(in_channels, c_channels, 1, bias=False)
        self.filter = conv(c_channels, out_channels, 3, bias=False)
        self.layer = layer

        self.init_iters = init_iters
        self.update_iters = update_iters
        self.filter_reg = filter_reg
        self.precond = precond
        self.direction_forget_factor = (1 - precond_lr) ** CG_forgetting_rate
        self.train_skipping = train_skipping
        self.learning_rate = learning_rate
        self.memory_size = memory_size

        self.pw_params = pixel_weighting
        self.device = device
        self.update_filters = update_filters
        self.to(device)

        self.frame_num = 0
        self.update_optimizer = None
        self.current_sample = None
        self.memory = None

    def _is_finite(self, t):
        return (torch.isnan(t) + torch.isinf(t)) == 0

    def compute_pixel_weights(self, y):
        """
        :param pixel_weighting:   dict(method={'fixed'|'hinged'}, tf=target influence (fraction), per_frame=bool)
        :param y:                 Training labels (tensor, shape = (N, 1, H, W)), N = number of samples
        :return:  tensor (N, 1, H, W) of pixel weights
        """
        if self.pw_params is None or self.pw_params['method'] == 'none':
            return torch.ones_like(y)

        method = self.pw_params['method']
        tf = self.pw_params['tf']  # Target influence of foreground pixels
        assert method == 'hinge'

        N, C, H, W = y.shape

        y = y.float()
        px = y.sum(dim=(2, 3))
        af = px / (H * W)
        px = px.view(N, C, 1, 1)
        af = af.view(N, C, 1, 1)

        too_small = (px < 10).float()
        af = too_small * tf + (1 - too_small) * af

        ii = (af > tf).float()        # if af > tf (i.e the object is large enough), then set tf = af
        tf = ii * af + (1 - ii) * tf  # this leads to self.pixel_weighting = 1 (i.e do nothing)

        wf = tf / af  # Foreground pixels weight
        wb = (1 - tf) / (1 - af)  # Background pixels weight

        training = False
        if training:
            invalid = ~self._is_finite(wf)
            if invalid.sum().item() > 0:
                print("Warning: Corrected invalid (non-finite) foreground filter pixel-weights.")
                wf[invalid] = 1.0

            invalid = ~self._is_finite(wb)
            if invalid.sum() > 0:
                print("Warning: Corrected bad (non-finite) background pixel-weights.")
                wb[invalid] = 1.0

        w = wf * y + wb * (1 - y)
        w = torch.sqrt(w)

        return w

    def init(self, x, y):
        """
        :param x: Tensor of data augmented features from the first frame, shape (K, Cf, Hf, Wf),
                  where K is the number of augmented feature maps.
        :param y: Object mask tensor, shape (K, 1, Him, Wim)
        """

        pw = self.compute_pixel_weights(y)

        # Run the initial optimization

        memory = Memory(y.shape[0], x.shape[-3:], y.shape[-3:], self.device, self.learning_rate)
        memory.initialize(x, y, pw)

        parameters = TensorList([self.project.weight, self.filter.weight])
        problem = DiscriminatorLoss(x=memory.samples, y=memory.labels,
                                    filter_regs=self.filter_reg, precond=self.precond, sample_weights=memory.weights,
                                    net=nn.Sequential(self.project, self.filter), pixel_weighting=memory.pixel_weights)
        optimizer = GaussNewtonCG(problem, parameters, fletcher_reeves=False, standard_alpha=True,
                                  direction_forget_factor=self.direction_forget_factor)
        problem.net.train()
        optimizer.run(self.init_iters)
        problem.net.eval()

        x = self.project(x)  # Re-project samples with the new projection matrix

        # Initialize the memory

        memory = Memory(self.memory_size, x.shape[-3:], y.shape[-3:], self.device, self.learning_rate)
        memory.initialize(x, y, pw)

        # Build the update problem

        parameters = TensorList([self.filter.weight])
        problem = DiscriminatorLoss(x=memory.samples, y=memory.labels,
                                    filter_regs=self.filter_reg[1:], precond=self.precond[1:],
                                    sample_weights=memory.weights, net=self.filter,
                                    pixel_weighting=memory.pixel_weights)
        optimizer = GaussNewtonCG(problem, parameters, fletcher_reeves=False, standard_alpha=True,
                                  direction_forget_factor=self.direction_forget_factor)
        problem.net.train()
        optimizer.run(self.update_iters)
        problem.net.eval()

        self.memory = memory
        self.update_optimizer = optimizer

    def apply(self, ft):
        self.frame_num += 1
        cft = self.project(ft)
        self.current_sample = cft
        scores = self.filter(cft)
        return scores

    def update(self, train_y):

        if not self.update_filters:
            return
        if self.current_sample is None:
            return
        if (train_y > 0.5).sum() < 10:
            return

        ys = (train_y > 0.5).float()
        pw = self.compute_pixel_weights(ys)
        self.memory.update(self.current_sample, train_y, pw)

        if self.frame_num % self.train_skipping != 0:
            return

        optim = self.update_optimizer
        optim.problem.net.train()
        optim.run(self.update_iters)
        optim.problem.net.eval()
