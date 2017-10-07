"""
Demonstrate supervised learning using zounds.  First, perform a
frequency-adaptive transform (where time-resolution varies by frequency)
of some bach pieces, and then rasterize these.

Use a neural network to learn to recover the original frequency-adaptive
transform from the rasterized version.
"""

import featureflow as ff
import numpy as np
import zounds
from torch import nn
from torch import optim
from random import choice

samplerate = zounds.SR11025()
BaseModel = zounds.stft(resample_to=samplerate, store_fft=True)

scale = zounds.GeometricScale(
    start_center_hz=300,
    stop_center_hz=3040,
    bandwidth_ratio=0.07496,
    n_bands=64)
scale.ensure_overlap_ratio(0.5)


@zounds.simple_lmdb_settings('bach', map_size=1e10, user_supplied_id=True)
class Sound(BaseModel):
    """
    An audio processing pipeline that computes a frequency domain representation
    of the sound that follows a geometric scale
    """
    bark = zounds.ArrayWithUnitsFeature(
        zounds.BarkBands,
        samplerate=samplerate,
        stop_freq_hz=samplerate.nyquist,
        needs=BaseModel.fft,
        store=True)

    long_windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=zounds.SampleRate(
            frequency=zounds.Milliseconds(358),
            duration=zounds.Milliseconds(716)),
        wfunc=zounds.OggVorbisWindowingFunc(),
        needs=BaseModel.resampled,
        store=True)

    long_fft = zounds.ArrayWithUnitsFeature(
        zounds.FFT,
        needs=long_windowed,
        store=True)

    freq_adaptive = zounds.FrequencyAdaptiveFeature(
        zounds.FrequencyAdaptiveTransform,
        transform=np.fft.irfft,
        scale=scale,
        window_func=np.hanning,
        needs=long_fft,
        store=False)

    rasterized = zounds.ArrayWithUnitsFeature(
        lambda fa: fa.rasterize(64),
        needs=freq_adaptive,
        store=False)


class Layer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(Layer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(2, 2),
            stride=(2, 2),
            bias=False)
        self.activation = nn.Tanh()

    def forward(self, inp):
        inp = self.conv(inp)
        inp = self.activation(inp)
        return inp


class UpScale(nn.Module):
    def __init__(self):
        super(UpScale, self).__init__()
        self.main = nn.Sequential(
            Layer(1, 4),
            Layer(4, 16),
            Layer(16, 64),
            Layer(64, 256),
            Layer(256, 1024),
            Layer(1024, 8192))

    def forward(self, inp):
        output = self.main(inp.view(-1, 1, 64, 64))
        return output.view(-1, 8192)


@zounds.simple_settings
class FeatureTransfer(ff.BaseModel):
    docs = ff.PickleFeature(
        ff.IteratorNode,
        store=False)

    shuffled = ff.PickleFeature(
        zounds.ShuffledSamples,
        nsamples=int(1e5),
        multiplexed=True,
        dtype=np.float32,
        needs=docs,
        store=False)

    mu_law_source = ff.PickleFeature(
        zounds.MuLawCompressed,
        needs=shuffled.aspect('data'),
        store=False)

    scaled_source = ff.PickleFeature(
        zounds.InstanceScaling,
        needs=mu_law_source,
        store=False)

    mu_law_target = ff.PickleFeature(
        zounds.MuLawCompressed,
        needs=shuffled.aspect('labels'),
        store=False)

    scaled_target = ff.PickleFeature(
        zounds.InstanceScaling,
        needs=mu_law_target,
        store=False)

    network = ff.PickleFeature(
        zounds.PyTorchNetwork,
        trainer=zounds.SupervisedTrainer(
            model=UpScale(),
            loss=nn.MSELoss(),
            optimizer=lambda model: optim.Adam(model.parameters()),
            epochs=20,
            batch_size=64,
            holdout_percent=0.5),
        needs=dict(data=scaled_source, labels=scaled_target),
        store=False)

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(mu_law_source, scaled_source, network),
        store=True)


if __name__ == '__main__':

    zounds.ingest(
        zounds.InternetArchive('AOC11B'),
        Sound,
        multi_threaded=True)

    if not FeatureTransfer.exists():
        FeatureTransfer.process(
            docs=(dict(data=doc.rasterized, labels=doc.freq_adaptive)
                  for doc in Sound))

    snds = list(Sound)
    snd = choice(snds)

    feature_transfer = FeatureTransfer()

    synth = zounds.FrequencyAdaptiveFFTSynthesizer(scale, samplerate)

    original = synth.synthesize(snd.freq_adaptive)

    recon_coeffs = feature_transfer\
        .pipeline\
        .transform(snd.rasterized, wrapper=snd.freq_adaptive.like_dims)\
        .data

    recon = synth.synthesize(zounds.inverse_mu_law(recon_coeffs))

    # start up an in-browser REPL to interact with the results
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.bark,
        globals=globals(),
        locals=locals())
    app.start(8888)
