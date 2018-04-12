from torch import nn
from zounds.spectral import fir_filter_bank
from scipy.signal import gaussian
from torch.autograd import Variable
import torch
from torch.nn import functional as F
from dct_transform import DctTransform
from zounds.timeseries import Picoseconds, SampleRate
import numpy as np


class PerceptualLoss(nn.MSELoss):
    def __init__(
            self,
            scale,
            samplerate,
            frequency_window=gaussian(100, 3),
            basis_size=512,
            lap=2,
            log_factor=100,
            frequency_weighting=None,
            phase_locking_cutoff_hz=None):

        super(PerceptualLoss, self).__init__()

        self.log_factor = log_factor
        self.scale = scale
        basis_size = basis_size
        self.lap = lap
        self.basis_size = basis_size

        basis = fir_filter_bank(
            scale, basis_size, samplerate, frequency_window)

        weights = Variable(torch.from_numpy(basis).float())
        # out channels x in channels x kernel width
        self.weights = weights.view(len(scale), 1, basis_size).contiguous()

        self.frequency_weights = None
        if frequency_weighting:
            fw = frequency_weighting._wdata(self.scale)
            self.frequency_weights = Variable(torch.from_numpy(fw).float())

        self.pool_amount = None
        if phase_locking_cutoff_hz is not None:
            sr = SampleRate(
                frequency=samplerate.frequency / lap,
                duration=samplerate.duration / lap)
            one_cycle = Picoseconds(int(1e12)) / phase_locking_cutoff_hz
            self.pool_amount = int(np.ceil(one_cycle / sr.frequency))

    def cuda(self, device=None):
        self.weights = self.weights.cuda()
        if self.frequency_weights is not None:
            self.frequency_weights = self.frequency_weights.cuda()
        return super(PerceptualLoss, self).cuda(device=device)

    def _transform(self, x):
        x = x.view(x.shape[0], 1, -1)
        features = F.conv1d(
            x, self.weights, stride=self.lap, padding=self.basis_size)

        # perceptual frequency weighting
        if self.frequency_weights is not None:
            features = \
                features * self.frequency_weights.view(1, len(self.scale), 1)

        # half-wave rectification
        features = F.relu(features)

        # log magnitude
        features = torch.log(1 + features * self.log_factor)

        # loss of phase locking
        if self.pool_amount is not None:
            features = features.view(x.shape[0], 1, len(self.scale), -1)
            features = F.max_pool2d(
                features, (1, self.pool_amount), stride=(1, 1))

        return features

    def forward(self, input, target):
        input = input.view(input.shape[0], 1, -1)
        target = target.view(input.shape[0], 1, -1)

        input_features = self._transform(input).view(input.shape[0], -1)
        target_features = self._transform(target).view(input.shape[0], -1)

        return -(F.cosine_similarity(input_features, target_features).mean())


class BandLoss(nn.MSELoss):
    def __init__(self, factors, spectral_shape_weight=1):
        super(BandLoss, self).__init__()
        self.spectral_shape_weight = spectral_shape_weight
        self.factors = factors
        self.dct_transform = DctTransform()

    def cuda(self, device=None):
        self.dct_transform = DctTransform(use_cuda=True)
        return super(BandLoss, self).cuda(device=device)

    def _transform(self, x):
        bands = self.dct_transform.frequency_decomposition(
            x, self.factors, axis=-1)
        maxes = [torch.max(b, dim=-1, keepdim=True)[0] for b in bands]
        bands = [b / n for (b, n) in zip(bands, maxes)]
        fine = torch.cat(bands, dim=-1)
        coarse = torch.cat(maxes, dim=-1)
        coarse_norms = torch.norm(coarse, dim=-1, keepdim=True)
        coarse = coarse / coarse_norms
        return fine, coarse

    def forward(self, input, target):
        input_bands, input_coarse = self._transform(input)
        target_bands, target_coarse = self._transform(target)
        fine = super(BandLoss, self).forward(input_bands, target_bands)
        coarse = super(BandLoss, self).forward(input_coarse, target_coarse)
        return fine + (coarse * self.spectral_shape_weight)


class CategoricalLoss(object):
    def __init__(self, n_categories):
        super(CategoricalLoss, self).__init__()
        self.n_categories = n_categories
        self.use_cuda = False
        self.loss = nn.NLLLoss2d()

    def cuda(self, device=None):
        self.use_cuda = True
        self.loss = self.loss.cuda(device=device)
        return self

    def _variable(self, x, *args, **kwargs):
        v = Variable(x, *args, **kwargs)
        if self.use_cuda:
            v = v.cuda()
        return v

    def _mu_law(self, x):
        m = self._variable(torch.FloatTensor(1))
        m[:] = self.n_categories + 1
        s = torch.sign(x)
        x = torch.abs(x)
        x = s * (torch.log(1 + (self.n_categories * x)) / torch.log(m))
        return x

    def _shift_and_scale(self, x):
        x = x + 1
        x = x * ((self.n_categories) / 2.)
        return x

    def _one_hot(self, x):
        y = self._variable(torch.arange(0, self.n_categories + 1))
        x = -(((x[..., None] - y) ** 2) * 1e12)
        x = F.log_softmax(x, dim=-1)
        return x

    def _discretized(self, x):
        x = x.view(-1)
        x = self._mu_law(x)
        x = self._shift_and_scale(x)
        return x

    def _categorical(self, x):
        x = self._discretized(x)
        x = self._one_hot(x)
        return x

    def forward(self, input, target):

        if input.shape[1] == self.n_categories + 1:
            categorical = input
        else:
            categorical = self._categorical(input)

        discretized = self._discretized(target)
        inp = categorical.view(
            -1, self.n_categories + 1, 2, input.shape[-1] // 2)
        t = discretized.view(-1, 2, target.shape[-1] // 2).long()
        error = self.loss(inp, t)
        return error

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class BaseLoss(object):
    def __init__(self):
        super(BaseLoss, self).__init__()
        self.use_cuda = False

    def _cuda(self, device=None):
        raise NotImplementedError()

    def cuda(self, device=None):
        self.use_cuda = True
        self._cuda(device=device)
        return self

    def forward(self, *args, **kwargs):
        raise NotImplementedError()

    def __call__(self, *args, **kwargs):
        return self.forward(*args, **kwargs)


class LearnedWassersteinLoss(BaseLoss):
    def __init__(self, critic):
        super(LearnedWassersteinLoss, self).__init__()
        self.critic = critic

    def _cuda(self, device=None):
        self.critic.cuda(device=device)

    def forward(self, x):
        w = self.critic(x)
        return -torch.mean(w)


class WassersteinCriticLoss(BaseLoss):
    def __init__(self, critic):
        super(WassersteinCriticLoss, self).__init__()
        self.critic = critic

    def _cuda(self, device=None):
        self.critic.cuda(device=device)

    def forward(self, real, fake):
        d_real = self.critic(real)
        d_fake = self.critic(fake)
        real_mean = torch.mean(d_real)
        fake_mean = torch.mean(d_fake)
        return fake_mean - real_mean


class WassersteinGradientPenaltyLoss(BaseLoss):
    def __init__(self, critic, weight=10):
        super(WassersteinGradientPenaltyLoss, self).__init__()
        self.weight = weight
        self.critic = critic

    def _cuda(self, device=None):
        self.critic.cuda(device=device)

    def forward(self, real_samples, fake_samples):
        from torch.autograd import grad

        real_samples = real_samples.view(fake_samples.shape)

        subset_size = real_samples.shape[0]

        real_samples = real_samples[:subset_size]
        fake_samples = fake_samples[:subset_size]

        alpha = torch.rand(subset_size)
        if self.use_cuda:
            alpha = alpha.cuda()
        alpha = alpha.view((-1,) + ((1,) * (real_samples.dim() - 1)))

        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        if self.use_cuda:
            interpolates = interpolates.cuda()
        interpolates = Variable(interpolates, requires_grad=True)

        d_output = self.critic(interpolates)

        output = torch.ones(d_output.size())
        if self.use_cuda:
            output = output.cuda()

        gradients = grad(
            outputs=d_output,
            inputs=interpolates,
            grad_outputs=output,
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * self.weight
