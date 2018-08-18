from torch import nn
import torch
import math
from torch.nn import functional as F


class SincLayer(nn.Module):
    """
    A layer as described in the paper
    "Speaker Recognition from raw waveform with SincNet"

    .. epigraph::

        This paper proposes a novel CNN architecture, called SincNet, that
        encourages the first convolutional layer to discover more meaningful
        filters. SincNet is based on parametrized sinc functions, which
        implement band-pass filters. In contrast to standard CNNs, that
        learn all elements of each filter, only low and high cutoff
        frequencies are directly learned from data with the proposed method.
        This offers a very compact and efficient way to derive a customized
        filter bank specifically tuned for the desired application. Our
        experiments, conducted on both speaker identification and speaker
        verification tasks, show that the proposed architecture converges
        faster and performs better than a standard CNN on raw waveforms.

        -- https://arxiv.org/abs/1808.00158


    Args:
        scale (FrequencyScale): A scale defining the initial bandpass
            filters
        taps (int): The length of the filter in samples
        samplerate (SampleRate): The sampling rate of incoming samples

    See Also:
        :class:`~zounds.spectral.FrequencyScale`
        :class:`~zounds.timeseries.SampleRate`
    """

    def __init__(self, scale, taps, samplerate):
        super(SincLayer, self).__init__()
        self.samplerate = int(samplerate)
        self.taps = taps
        self.scale = scale

        # each filter requires two parameters to define the filter bandwidth
        filter_parameters = torch.FloatTensor(len(scale), 2)

        self.linear = nn.Parameter(
            torch.linspace(-math.pi, math.pi, steps=taps), requires_grad=False)
        self.window = nn.Parameter(
            torch.hamming_window(self.taps), requires_grad=False)

        for i, band in enumerate(scale):
            start = self.samplerate / band.start_hz
            stop = self.samplerate / band.stop_hz
            filter_parameters[i, 0] = start
            filter_parameters[i, 1] = stop

        self.filter_parameters = nn.Parameter(filter_parameters)

    def _sinc(self, frequency):
        x = self.linear[None, :] * frequency[:, None]
        return torch.sin(x) / x

    def _start_frequencies(self):
        return torch.abs(self.filter_parameters[:, 0])

    def _stop_frequencies(self):
        start = self._start_frequencies()
        return start + torch.abs(self.filter_parameters[:, 1] - start)

    def _filter_bank(self):
        start = self._start_frequencies()[:, None]
        stop = self._stop_frequencies()[:, None]
        start_sinc = self._sinc(start)
        stop_sinc = self._sinc(stop)
        filters = \
            (2 * stop[..., None] * stop_sinc) \
            - (2 * start[..., None] * start_sinc)
        windowed = filters * self.window[None, None, :]
        return windowed.squeeze()

    def forward(self, x):
        x = x.view(-1, 1, x.shape[-1])
        filters = self._filter_bank().view(len(self.scale), 1, self.taps)
        x = F.conv1d(x, filters, stride=1, padding=self.taps // 2)
        return x
