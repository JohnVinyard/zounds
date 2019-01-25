import torch
from torch import nn
from torch.nn import functional as F
from zounds.spectral import morlet_filter_bank, AWeighting, FrequencyDimension
from zounds.core import ArrayWithUnits
from zounds.timeseries import TimeDimension


class FilterBank(nn.Module):
    def __init__(self, samplerate, kernel_size, scale, scaling_factors):
        super(FilterBank, self).__init__()

        filter_bank = morlet_filter_bank(
            samplerate,
            kernel_size,
            scale,
            scaling_factors)
        filter_bank *= AWeighting()

        self.scale = scale
        self.filter_bank = torch.from_numpy(filter_bank).float() \
            .view(len(scale), 1, kernel_size)
        self.filter_bank.requires_grad = False

    def to(self, *args, **kwargs):
        self.filter_bank = self.filter_bank.to(*args, **kwargs)
        return super(FilterBank, self).to(*args, **kwargs)

    def convolve(self, x):
        x = x.view(-1, 1, x.shape[-1])
        x = F.conv1d(
            x, self.filter_bank, padding=self.filter_bank.shape[-1] // 2)
        return x

    def transposed_convolve(self, x):
        x = F.conv_transpose1d(
            x, self.filter_bank, padding=self.filter_bank.shape[-1] // 2)
        return x

    def log_magnitude(self, x):
        x = F.relu(x)
        x = 20 * torch.log10(1 + x)
        return x

    def temporal_pooling(self, x, kernel_size, stride):
        x = F.avg_pool1d(x, kernel_size, stride, padding=kernel_size // 2)
        return x

    def normalize(self, x):
        """
        give each instance zero mean and unit variance
        """
        orig_shape = x.shape
        x = x.view(x.shape[0], -1)
        x = x - x.mean(dim=1, keepdim=True)
        x = x / (x.std(dim=1, keepdim=True) + 1e-8)
        x = x.view(orig_shape)
        return x

    def transform(self, samples, pooling_kernel_size, pooling_stride):
        # convert the raw audio samples to a PyTorch tensor
        tensor_samples = torch.from_numpy(samples).float() \
            .to(self.filter_bank.device)

        # compute the transform
        spectral = self.convolve(tensor_samples)
        log_magnitude = self.log_magnitude(spectral)
        pooled = self.temporal_pooling(
            log_magnitude, pooling_kernel_size, pooling_stride)

        # convert back to an ArrayWithUnits instance
        samplerate = samples.samplerate
        time_frequency = pooled.data.cpu().numpy().squeeze().T
        time_frequency = ArrayWithUnits(time_frequency, [
            TimeDimension(
                frequency=samplerate.frequency * pooling_stride,
                duration=samplerate.frequency * pooling_kernel_size),
            FrequencyDimension(self.scale)
        ])
        return time_frequency

    def forward(self, x, normalize=True):
        nsamples = x.shape[-1]
        x = self.convolve(x)
        x = self.log_magnitude(x)

        if normalize:
            x = self.normalize(x)

        return x[..., :nsamples].contiguous()