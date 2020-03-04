import torch
from torch import nn
from torch.nn import functional as F
from zounds.spectral import morlet_filter_bank, AWeighting, FrequencyDimension
from zounds.core import ArrayWithUnits
from zounds.timeseries import TimeDimension
from .util import batchwise_mean_std_normalization


class FilterBank(nn.Module):
    """
    A torch module that convolves a 1D input signal with a bank of morlet
    filters.

    Args:
        samplerate (SampleRate): the samplerate of the input signal
        kernel_size (int): the length in samples of each filter
        scale (FrequencyScale): a scale whose center frequencies determine the
            fundamental frequency of each filer
        scaling_factors (int or list of int): Scaling factors for each band,
            which determine the time-frequency resolution tradeoff.
            The number(s) should fall between 0 and 1, with smaller numbers
            achieving better frequency resolution, and larget numbers better
            time resolution
        normalize_filters (bool): When true, ensure that each filter in the bank
            has unit norm
        a_weighting (bool): When true, apply a perceptually-motivated weighting
            of the filters

    See Also:
        :class:`~zounds.spectral.AWeighting`
        :func:`~zounds.spectral.morlet_filter_bank`
    """

    def __init__(
            self,
            samplerate,
            kernel_size,
            scale,
            scaling_factors,
            normalize_filters=True,
            a_weighting=True):

        super(FilterBank, self).__init__()

        self.samplerate = samplerate
        filter_bank = morlet_filter_bank(
            samplerate,
            kernel_size,
            scale,
            scaling_factors,
            normalize=normalize_filters)

        if a_weighting:
            filter_bank *= AWeighting()

        self.scale = scale

        filter_bank = torch.from_numpy(filter_bank).float() \
            .view(len(scale), 1, kernel_size)
        self.register_buffer('filter_bank', filter_bank)

    @property
    def n_bands(self):
        return len(self.scale)

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
            x = batchwise_mean_std_normalization(x)

        return x[..., :nsamples].contiguous()
