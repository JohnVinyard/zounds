from torch import nn
from zounds.spectral import fir_filter_bank
from scipy.signal import gaussian
from torch.autograd import Variable
import torch
from torch.nn import functional as F
from dct_transform import DctTransform


class PerceptualLoss(nn.MSELoss):
    def __init__(
            self,
            scale,
            samplerate,
            frequency_window=gaussian(100, 3),
            basis_size=512,
            lap=2,
            log_factor=100):
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

    def cuda(self, device=None):
        self.weights = self.weights.cuda()
        return super(PerceptualLoss, self).cuda(device=device)

    def _transform(self, x):
        features = F.conv1d(
            x, self.weights, stride=self.lap, padding=self.basis_size)

        # half-wave rectification
        features = F.relu(features)

        # log magnitude
        features = torch.log(1 + features * self.log_factor)

        return features

    def forward(self, input, target):
        input = input.view(input.shape[0], 1, -1)
        target = target.view(input.shape[0], 1, -1)

        input_features = self._transform(input)
        target_features = self._transform(target)

        return super(PerceptualLoss, self).forward(
            input_features, target_features)


class BandLoss(nn.MSELoss):
    def __init__(self, factors):
        super(BandLoss, self).__init__()
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
        return torch.cat(bands, dim=-1)

    def forward(self, input, target):
        input_bands = self._transform(input)
        target_bands = self._transform(target)
        return super(BandLoss, self).forward(input_bands, target_bands)


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
