import featureflow as ff
import numpy as np
import zounds
from torch import nn
from torch import optim
from multiprocessing.pool import ThreadPool, cpu_count

samplerate = zounds.SR11025()
BaseModel = zounds.stft(resample_to=samplerate, store_fft=True)

scale = zounds.GeometricScale(
    start_center_hz=300,
    stop_center_hz=3040,
    bandwidth_ratio=0.07496,
    n_bands=64)
scale.ensure_overlap_ratio(0.5)

LATENT_DIM = 100


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


class GeneratorLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GeneratorLayer, self).__init__()
        self.conv = nn.ConvTranspose2d(
            in_features,
            out_features,
            kernel_size=(2, 2),
            stride=(2, 2),
            bias=False)
        self.batch_norm = nn.BatchNorm2d(out_features)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

    def forward(self, inp):
        x = self.conv(inp)
        x = self.batch_norm(x)
        x = self.relu(x)
        x = self.dropout(x)
        return x


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            GeneratorLayer(LATENT_DIM, 512),
            GeneratorLayer(512, 256),
            GeneratorLayer(256, 128),
            GeneratorLayer(128, 64),
            GeneratorLayer(64, 32),
            nn.ConvTranspose2d(32, 1, (2, 2), (2, 2)),
            nn.Tanh()
        )

    def forward(self, inp):
        return self.main(inp.view(-1, LATENT_DIM, 1, 1))


class DiscriminatorLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DiscriminatorLayer, self).__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=(2, 2),
            stride=(2, 2),
            bias=False)
        self.batch_norm = nn.BatchNorm2d(out_channels)
        self.relu = nn.LeakyReLU(0.2)
        self.dropout = nn.Dropout(0.2)

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


@zounds.simple_settings
class GanPipeline(ff.BaseModel):
    docs = ff.PickleFeature(
        ff.IteratorNode,
        store=False)

    shuffled = ff.PickleFeature(
        zounds.ShuffledSamples,
        nsamples=int(1e5),
        dtype=np.float32,
        needs=docs,
        store=False)

    mu_law = ff.PickleFeature(
        zounds.MuLawCompressed,
        needs=shuffled,
        store=False)

    scaled = ff.PickleFeature(
        zounds.InstanceScaling,
        needs=mu_law,
        store=False)

    network = ff.PickleFeature(
        zounds.PyTorchGan,
        trainer=zounds.GanTrainer(
            generator=Generator(),
            discriminator=Discriminator(),
            loss=nn.BCELoss(),
            generator_optim_func=lambda model: optim.Adam(
                model.parameters(), lr=0.0002, betas=(0.5, 0.999)),
            discriminator_optim_func=lambda model: optim.Adam(
                model.parameters(), lr=0.00005, betas=(0.5, 0.999)),
            latent_dimension=(LATENT_DIM,),
            epochs=500,
            batch_size=64),
        needs=scaled,
        store=False)

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(mu_law, scaled, network),
        store=True)


if __name__ == '__main__':

    source = zounds.InternetArchive('AOC11B')
    for metadata in source:
        request = metadata.request
        url = request.url
        if Sound.exists(request.url):
            print 'already processed {request.url}'.format(**locals())
            continue

        print 'processing {request.url}'.format(**locals())
        Sound.process(meta=request, _id=request.url)

    if not GanPipeline.exists():
        GanPipeline.process(docs=(snd.rasterized for snd in Sound))

    gan = GanPipeline()
    noise = np.random.normal(0, 1, (32, LATENT_DIM))
    generated_samples = gan.pipeline.transform(noise)

    # start up an in-browser REPL to interact with the results
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.bark,
        globals=globals(),
        locals=locals())
    app.start(8888)
