import featureflow as ff
import zounds
import numpy as np
from torch import nn, optim
from random import choice
import argparse


class Layer(nn.Module):
    """
    A single layer of our simple autoencoder
    """

    def __init__(self, in_size, out_size):
        super(Layer, self).__init__()
        self.linear = nn.Linear(in_size, out_size, bias=False)
        self.tanh = nn.Tanh()

    def forward(self, inp):
        x = self.linear(inp)
        x = self.tanh(x)
        return x


class AutoEncoder(nn.Module):
    """
    A simple autoencoder.  No bells, whistles, or convolutions
    """

    def __init__(self):
        super(AutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            Layer(8192, 1024),
            Layer(1024, 512),
            Layer(512, 256))
        self.decoder = nn.Sequential(
            Layer(256, 512),
            Layer(512, 1024),
            Layer(1024, 8192))

    def forward(self, inp):
        encoded = self.encoder(inp)
        decoded = self.decoder(encoded)
        return decoded


@zounds.simple_settings
class FreqAdaptiveAutoEncoder(ff.BaseModel):
    """
    Define a processing pipeline to learn a compressed representation of the
    Sound.freq_adaptive feature.  Once this is trained and the pipeline is
    stored, we can apply all the pre-processing steps and the autoencoder
    forward and in reverse.
    """
    docs = ff.Feature(ff.IteratorNode)

    shuffle = ff.PickleFeature(
        zounds.ShuffledSamples,
        nsamples=500000,
        dtype=np.float32,
        needs=docs)

    mu_law = ff.PickleFeature(
        zounds.MuLawCompressed,
        needs=shuffle)

    scaled = ff.PickleFeature(
        zounds.InstanceScaling,
        needs=mu_law)

    autoencoder = ff.PickleFeature(
        zounds.PyTorchAutoEncoder,
        trainer=zounds.SupervisedTrainer(
            AutoEncoder(),
            loss=nn.MSELoss(),
            optimizer=lambda model: optim.Adam(model.parameters(), lr=0.00005),
            epochs=100,
            batch_size=64,
            holdout_percent=0.5),
        needs=scaled)

    # assemble the previous steps into a re-usable pipeline, which can perform
    # forward and backward transformations
    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(mu_law, scaled, autoencoder),
        store=True)


samplerate = zounds.SR11025()
BaseModel = zounds.stft(resample_to=samplerate, store_fft=True)

scale = zounds.GeometricScale(
    start_center_hz=300,
    stop_center_hz=3040,
    bandwidth_ratio=0.016985,
    n_bands=300)
scale.ensure_overlap_ratio(0.5)


@zounds.simple_lmdb_settings('sounds', map_size=1e10, user_supplied_id=True)
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
            frequency=zounds.Milliseconds(340),
            duration=zounds.Milliseconds(680)),
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

    encoded = zounds.ArrayWithUnitsFeature(
        zounds.Learned,
        learned=FreqAdaptiveAutoEncoder(),
        needs=freq_adaptive,
        store=False)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--internet-archive-id',
        help='the internet archive id to process',
        type=str,
        required=False,
        default='AOC11B')
    args = parser.parse_args()

    zounds.ingest(
        dataset=zounds.InternetArchive(args.internet_archive_id),
        cls=Sound,
        multi_threaded=True)

    # train the pipeline, including the autoencoder
    if not FreqAdaptiveAutoEncoder.exists():
        FreqAdaptiveAutoEncoder.process(
            docs=(snd.freq_adaptive for snd in Sound))

    # get a reference to the trained pipeline
    autoencoder = FreqAdaptiveAutoEncoder()

    # get references to all the sounds.  features are lazily
    # loaded/evaluated, so this is a cheap operation
    snds = list(Sound)

    # create a synthesizer that can invert the frequency adaptive representation
    synth = zounds.FrequencyAdaptiveFFTSynthesizer(scale, samplerate)


    def random_reconstruction(random_encoding=False):
        # choose a random sound
        snd = choice(snds)

        # run the model forward
        encoded = autoencoder.pipeline.transform(snd.freq_adaptive)

        if random_encoding:
            mean = encoded.data.mean()
            std = encoded.data.std()
            encoded.data = np.random.normal(mean, std, encoded.data.shape)

        # then invert the encoded version
        inverted = encoded.inverse_transform()

        # compare the audio of the original and the reconstruction
        original = synth.synthesize(snd.freq_adaptive)
        recon = synth.synthesize(inverted)
        return original, recon, encoded.data, snd.freq_adaptive, inverted


    # get the original audio, and the reconstructed audio
    orig_audio, recon_audio, encoded, orig_coeffs, inverted_coeffs = \
        random_reconstruction()

    # start up an in-browser REPL to interact with the results
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.bark,
        globals=globals(),
        locals=locals())
    app.start(8888)
