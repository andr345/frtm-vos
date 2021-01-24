from collections import OrderedDict as odict
from easydict import EasyDict as edict
import torch
import torch.nn as nn

from .tensorlist import TensorList
from .utils import interpolate, conv
from .optimizer import MinimizationProblem, GaussNewtonCG
from .memory import Memory


class TargetModel(nn.Module):
    def __init__(self, input_channels, cdim, n_channels, ksz, with_bias=False):
        super().__init__()

        assert len(n_channels) == 1

        self.project = conv(input_channels, cdim, 1, bias=with_bias)
        self.filter = conv(cdim, n_channels[0], ksz[0], bias=with_bias)

    def forward(self, x):
        x = self.project(x)
        x = self.filter(x)
        return x


class DiscriminatorLoss(MinimizationProblem):

    def __init__(self, training_samples, training_labels, filter_regs, precon, sample_weights,
                 net, pixel_weighting, compute_norm=False):

        self.training_samples = training_samples
        self.y = training_labels
        self.y_size = training_labels.shape[-2:]
        self.filter_regs = filter_regs
        if sample_weights.size()[0] == 1:
            self.sample_weights = training_samples.new_zeros(training_samples.shape[0])
            self.sample_weights.fill_(sample_weights[0])
        else:
            self.sample_weights = sample_weights

        self.diag_M = precon
        self.pixel_weighting = pixel_weighting

        self.net = net
        self.compute_norm = compute_norm

    def __call__(self, x: TensorList):

        a = self.sample_weights > 0.0
        s = self.net(self.training_samples[a])
        s = interpolate(s, self.y_size)
        residuals = self.pixel_weighting[a] * (s - self.y[a])

        if self.sample_weights is not None:
            residuals = self.sample_weights[a].sqrt().view(-1, 1, 1, 1) * residuals

        residuals = TensorList([residuals])
        residuals.extend(self.filter_regs * x)

        if self.compute_norm:
            residuals = self.fast_ip(residuals, residuals)

        return residuals

    @staticmethod
    def fast_ip(a, b):
        return a.view(-1) @ b.view(-1)

    def ip_input(self, a: TensorList, b: TensorList):
        out = self.fast_ip(a, b)
        sum_out = sum(o.unsqueeze(dim=0) for o in out)
        rep_sum = TensorList([])
        for o in out:
            rep_sum.extend(sum_out.clone())

        return rep_sum

    def ip_output(self, a: TensorList, b: TensorList):
        out = self.fast_ip(a, b)
        return sum(o for o in out)

    def ip_debug(self, a: TensorList, b: TensorList):
        out = self.fast_ip(a, b)
        sum_out = sum(o for o in out)
        return out[0], sum_out

    def M1(self, x: TensorList):
        return x / self.diag_M


class Discriminator:

    def layer_index(self, layer_names):
        """ Function defining the layer order in the TensorList objects """
        return list(sorted(layer_names))

    def tensor_dict_to_tensor_list(self, tdict):
        return TensorList([tdict[L] for L in self.layer_index(tdict.keys())])

    def tensor_list_to_tensor_dict(self, tlist, layer_names):
        return odict([(L, t) for L, t in zip(self.layer_index(layer_names), tlist)])

    def __init__(self, params):
        self.params = params

        # Parameters

        self.feature_names = [self.params.layer]
        self.compressed_dim = []
        self.kernel_size = []
        self.n_channels = []
        self.filter_reg = []
        self.precon = []

        self.with_bias = []
        self.init_iters = []
        self.update_iters = []

        if 'pixel_weighting' not in params:
            self.pixel_weighting_params = None
        else:
            self.pixel_weighting_params = params.pixel_weighting

        p = self.params
        self.compressed_dim.append(p.cdims)
        self.kernel_size.append(p.kernel_size)
        self.filter_reg.append(TensorList(p.filter_reg))
        self.precon.append(TensorList(p.precon))

        self.n_channels.append(p.n_channels)
        self.with_bias.append(p.with_bias)
        self.init_iters.append(p.init_iters)
        self.update_iters.append(p.update_iters)

        if 'clamp_output' not in params:
            self.clamp_output = False
        else:
            self.clamp_output = params.clamp_output

        self.feature_sz = None
        self.networks = None

        # Memory

        self.init_sample_weights = None
        self.init_training_samples = None
        self.init_training_labels = None
        self.training_samples = None
        self.num_stored_samples = 0
        self.current_sample = None
        self.memory = None

        if 'updated_pretrained_filters' not in params:
            self.updated_pretrained_filters = False
        else:
            self.updated_pretrained_filters = params.updated_pretrained_filters

    def _is_finite(self, t):
        return (torch.isnan(t) + torch.isinf(t)) == 0

    def _compute_pixel_weights(self, pixel_weighting, y):
        """
        :param pixel_weighting:   dict(method={'fixed'|'hinged'}, tf=target influence (fraction), per_frame=bool)
        :param y:                 Training labels (tensor, shape = (N, 1, H, W)), N = number of samples
        :return:  tensor (N, 1, H, W) of pixel weights
        """
        y = y.float()
        distractors = (y > 1)
        w = y.new_ones((y.shape[0], y.shape[1], 1, 1))

        if len(pixel_weighting) == 0:
            return w

        method = pixel_weighting.method
        per_frame = pixel_weighting.per_frame

        if method in ('fixed', 'hinge','first-frame'):

            tf = pixel_weighting.tf  # Target influence of foreground pixels
            N, C, H, W = y.shape

            if per_frame:
                px = y.sum(dim=(2, 3))
                af = px / (H * W)
                px = px.reshape((N, C, 1, 1))
                af = af.reshape((N, C, 1, 1))
            else:
                px = y.sum(dim=(0, 2, 3))
                af = px / y.numel()  # Actual influence

            too_small = (px < 10).float()
            af = too_small * tf + (1 - too_small) * af

            if method == 'hinge':
                # if af > tf (i.e the object is large enough), then set tf = af
                # this leads to self.pixel_weighting = 1 (i.e do nothing)
                ii = (af > tf).float()
                tf = ii * af + (1 - ii) * tf
            elif method == 'first-frame':
                af[1:] = af[0]
                ii = (af > tf).float()
                tf = ii * af + (1 - ii) * tf

            wf = tf / af  # Foreground pixels weight

            invalid = ~self._is_finite(wf)
            if invalid.sum().item() > 0:
                print("Warning: Corrected invalid (non-finite) foreground filter pixel-weights.")
                wf[invalid] = 1.0

            wb = (1 - tf) / (1 - af)  # Background pixels weight

            invalid = ~self._is_finite(wb)
            if invalid.sum() > 0:
                print("Warning: Corrected bad (non-finite) background pixel-weights.")
                wb[invalid] = 1.0

            w = wf * y + wb * (1 - y)
            w[distractors] *= pixel_weighting.distractor_mult
            w = torch.sqrt(w)

        return w

    def compute_pixel_weights(self, params, training_labels):

        if params is None or params.method == 'none':
            return torch.ones(training_labels.shape)

        if isinstance(params, (list, tuple)):  # One setting per layer

            pixel_weighting = TensorList()

            for pw, sz in zip(params, self.feature_sz):
                pw = self._compute_pixel_weights(edict(pw), training_labels)
                pixel_weighting.append(pw)

        else:
            # Common setting for all layers
            pw = self._compute_pixel_weights(edict(params), training_labels)
            pixel_weighting = pw

        return pixel_weighting

    def setup_problem(self, networks, memory, filter_regs, precons, compute_norm=False):

        parameters = []
        for m in networks:
            parameters.append(TensorList([ff for ff in m.parameters()]))

        problems = [DiscriminatorLoss(
            x, lb, fr, pc, sw, m, pixel_weighting=pw, compute_norm=compute_norm)
            for x, lb, sw, pw, m, fr, pc in zip(memory.samples, memory.labels, memory.weights, memory.pixel_weights, networks, filter_regs, precons)]

        p = self.params
        optimizers = [GaussNewtonCG(problem, parms, debug=(p.debug >= 2)) for problem, parms in zip(problems, parameters)]
        return optimizers

    def init(self, xs, y):
        """
        :param xs: Dict (layer_name: data tensor) of data augmented features from the first frame.
                   Each element in xs has shape (K, Ci, Hi, Wi) where K is the
                   number of augmented feature maps.
        :param y: Object mask tensor, shape (K, 1, H1, W1)
        :return:
        """

        self.frame_num = 0

        p = self.params

        xs = self.tensor_dict_to_tensor_list(xs)
        self.feature_sz = TensorList([x.shape[-3:] for x in xs])

        # Optimization options

        p.precond_learning_rate = TensorList([self.params.learning_rate])
        if p.CG_forgetting_rate is None or max(p.precond_learning_rate) >= 1:
            p.direction_forget_factor = 0
        else:
            p.direction_forget_factor = (1 - max(p.precond_learning_rate)) ** p.CG_forgetting_rate

        lrs = [self.params.learning_rate]

        # Build the init problem

        self.memory = Memory(y.shape[0], self.feature_sz, y.shape[-3:], self.params.device, lrs)
        pixel_weighting = self.compute_pixel_weights(self.pixel_weighting_params, y)
        self.memory.initialize(xs, (y == 1).byte(), pixel_weighting)

        self.networks = [TargetModel(x.shape[1], cdim, nch, ksz, with_bias=bias).to(self.params.device)
                         for x, cdim, nch, ksz, bias in
                         zip(xs, self.compressed_dim, self.n_channels, self.kernel_size, self.with_bias)]

        self.filter_reg = []
        self.precon = []
        for m in self.networks:
            self.precon.append(p.precon)
            self.filter_reg.append(p.filter_reg)

        init_optimizers = self.setup_problem(self.networks, self.memory, self.filter_reg, self.precon, p.compute_norm)
        for opt, ip in zip(init_optimizers, self.init_iters):
            opt.run(ip)

        # Re-project samples with the new projection matrix

        compressed_samples = self.project_sample(xs)
        self.feature_sz = TensorList([x.shape[-3:] for x in compressed_samples])
        self.memory = Memory(self.params.sample_memory_size, self.feature_sz, y.shape[-3:], self.params.device, lrs)
        self.memory.initialize(compressed_samples, y == 1, pixel_weighting)

        # Build the update problem

        nns = [m.filter for m in self.networks]
        fregs = [r[1:] for r in self.filter_reg]
        precons = [r[1:] for r in self.precon]
        self.update_optimizers = self.setup_problem(nns, self.memory, fregs, precons, p.compute_norm)

        for f_opt, ip in zip(self.update_optimizers, self.update_iters):
            f_opt.run(ip)

        for ff in self.networks:
            ff.eval()

    def clear_memory(self):
        del self.memory
        self.memory = None

    def preprocess_sample(self, x):
        self.frame_num += 1  # Fix this
        x = self.tensor_dict_to_tensor_list(x)
        x = self.project_sample(x)
        x = self.tensor_list_to_tensor_dict(x, self.feature_names)
        return x

    def apply(self, features):
        self.current_sample = features
        features = self.tensor_dict_to_tensor_list(features)
        scores = []
        for f, x in zip(self.networks, features):
            s = f.filter(x)
            if self.clamp_output:
                s = s.clamp(-0.1, 1.2)
            scores.append(s)
        scores = self.tensor_list_to_tensor_dict(scores, self.feature_names)
        return scores

    def project_sample(self, features: TensorList):
        return TensorList([f.project(x) for x, f in zip(features, self.networks)])

    def update(self, train_y):

        if not self.params.update_filters:
            return
        if self.current_sample is None:
            return
        if (train_y > 0.5).sum() < 10:
            return

        pixel_weighting, ys = self.get_online_weights(train_y)
        self.memory.update(self.tensor_dict_to_tensor_list(self.current_sample), train_y, pixel_weighting)

        # Train filter
        if self.frame_num % self.params.train_skipping == 0:
            for ff in self.networks:
                ff.train()

            for opt, ip, ftname in zip(self.update_optimizers, self.update_iters, self.feature_names):
                ext_losses, int_losses, residuals = opt.run(ip)

        for ff in self.networks:
            ff.eval()

    def get_online_weights(self, sample_y):

        if self.pixel_weighting_params.update_method == "thresh":
            y = (sample_y > 0.5).float()
            pw = self.compute_pixel_weights(self.pixel_weighting_params, y)
            return pw, y
        elif self.pixel_weighting_params.update_method == "conf":
            pw = 2*(0.5-sample_y).abs()
            y = (sample_y > 0.5).float()
            pw2 = self.compute_pixel_weights(self.pixel_weighting_params, y)
            conf = pw.sqrt() * pw2
            return conf, y
        elif self.pixel_weighting_params.update_method == "raw":
            pw = sample_y.new_ones(sample_y.size(), dtype=torch.float)
            return pw, sample_y
        elif self.pixel_weighting_params.update_method == "raw-conf":
            pw = 2*(sample_y-0.5).abs()
            return pw, sample_y
        else:
            pw = sample_y.new_ones(sample_y.size(), dtype=torch.float)
            return pw, sample_y

    def A(self, x, dfdxt_g, g, f0, xn):
        dfdx_x = torch.autograd.grad(dfdxt_g, g, x, retain_graph=True)
        return TensorList(torch.autograd.grad(f0, xn, dfdx_x, retain_graph=True))
