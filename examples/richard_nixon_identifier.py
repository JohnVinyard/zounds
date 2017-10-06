import featureflow as ff
import numpy as np
import zounds
from torch import nn
from torch import optim
import argparse
from multiprocessing.pool import ThreadPool, cpu_count

samplerate = zounds.SR11025()
BaseModel = zounds.stft(resample_to=samplerate, store_fft=True)

scale = zounds.GeometricScale(
    start_center_hz=300,
    stop_center_hz=3040,
    bandwidth_ratio=0.07496,
    n_bands=64)
scale.ensure_overlap_ratio(0.5)


@zounds.simple_lmdb_settings('speeches', map_size=1e10, user_supplied_id=True)
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


class DiscriminatorLayer(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size=(2, 2),
            stride=None):

        super(DiscriminatorLayer, self).__init__()

        if stride is None:
            stride = kernel_size

        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            stride=stride,
            bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, inp):
        x = self.conv(inp)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            DiscriminatorLayer(1, 32),
            DiscriminatorLayer(32, 64),
            DiscriminatorLayer(64, 128),
            DiscriminatorLayer(128, 256),
            DiscriminatorLayer(256, 512),
            nn.Conv2d(512, 1, (2, 2), (2, 2), bias=False),
            nn.Sigmoid())

    def forward(self, inp):
        return self.main(inp.view(-1, 1, 64, 64))


def speaker_identification_pipeline(epochs):
    @zounds.simple_settings
    class RichardNixonIdentifier(ff.BaseModel):
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

        network = ff.PickleFeature(
            zounds.PyTorchNetwork,
            trainer=zounds.SupervisedTrainer(
                model=Discriminator(),
                loss=nn.BCELoss(),
                optimizer=lambda model:
                optim.Adam(model.parameters(), lr=0.00005),
                epochs=epochs,
                batch_size=64,
                holdout_percent=0.5),
            needs=dict(data=scaled_source, labels=shuffled.aspect('labels')),
            store=False)

        pipeline = ff.PickleFeature(
            zounds.PreprocessingPipeline,
            needs=(mu_law_source, scaled_source, network),
            store=True)

    return RichardNixonIdentifier


def process(metadata):

    request = metadata.request
    url = request.url

    if Sound.exists(url):
        print 'already processed {url}'.format(**locals())
        return

    print 'processing {url}'.format(**locals())
    Sound.process(meta=metadata, _id=url)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--epochs',
        help='how many epochs (full passes over data) should the network train',
        default=100,
        type=int)
    parser.add_argument(
        '--force',
        help='retrain the network, even if its already been trained',
        action='store_true',
        default=False)

    args = parser.parse_args()

    source = zounds.InternetArchive('Greatest_Speeches_of_the_20th_Century')

    pool = ThreadPool(cpu_count())
    _ids = pool.map(process, source)

    def generate_training_and_test_set():
        snds = list(Sound)

        # get all sounds where Nixon is the speaker
        nixon = filter(lambda snd: 'Nixon' in snd.meta['artist'], snds)

        # get an equal number of speeches by anyone besides Nixon
        not_nixon = filter(
            lambda snd: 'Nixon' not in snd.meta['artist'], snds)[:len(nixon)]

        for snd in nixon:
            yield dict(
                data=snd.rasterized,
                labels=np.ones((len(snd.rasterized), 1)))

        for snd in not_nixon[:len(nixon)]:
            yield dict(
                data=snd.rasterized,
                labels=np.zeros((len(snd.rasterized), 1)))


    RichardNixonIdentifier = speaker_identification_pipeline(args.epochs)

    if not RichardNixonIdentifier.exists() or args.force:
        RichardNixonIdentifier.process(docs=generate_training_and_test_set())

    rni = RichardNixonIdentifier()

    # start up an in-browser REPL to interact with the results
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.bark,
        globals=globals(),
        locals=locals())
    app.start(8888)
