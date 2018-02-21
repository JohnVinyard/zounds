from __future__ import division
import numpy as np
import featureflow as ff
import zounds
from torch import nn
import torch
from torch.autograd import Variable
import argparse
import glob
import os
from pytorch_wgan2 import \
    BaseGenerator, BaseCritic, CriticLayer, GeneratorLayer, FinalGeneratorLayer
from scipy.signal import resample, tukey
from random import choice
import torch.nn.functional as F
from uuid import uuid4

samplerate = zounds.SR11025()
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

LATENT_DIM = 100
SAMPLE_SIZE = 8192
bands = (8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096)
FIRST_FEATURE_MAP_SIZE = 64
FACTOR = 1.2

stops = tuple(np.cumsum(bands))
slices = [slice(start, stop) for (start, stop) in zip((0,) + stops, stops)]

SCALING = [
    0.035643139891883342,
    0.041599504468638721,
    0.043825312492803623,
    0.089081457396139319,
    0.11216649030248733,
    0.1755375826822119,
    0.3011956255933676,
    0.50373631894723525,
    0.72654767098659556,
    1.0668680716129715
]


def perceptual(x):
    coeffs = np.fft.rfft(x, norm='ortho', axis=-1)
    scale = zounds.LinearScale.from_sample_rate(samplerate, coeffs.shape[-1])
    arr = zounds.ArrayWithUnits(
        coeffs, [x.dimensions[0], zounds.FrequencyDimension(scale)])
    arr *= zounds.AWeighting()
    samples = np.fft.irfft(arr, norm='ortho', axis=-1)
    return zounds.ArrayWithUnits(samples, x.dimensions)


@zounds.simple_lmdb_settings('wgan', map_size=1e10, user_supplied_id=True)
class Sound(BaseModel):
    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=zounds.SampleRate(
            frequency=samplerate.frequency * (SAMPLE_SIZE // 2),
            duration=samplerate.frequency * SAMPLE_SIZE),
        needs=BaseModel.resampled,
        store=False)

    perceptual = zounds.ArrayWithUnitsFeature(
        perceptual,
        needs=windowed)

    decomposed = zounds.ArrayWithUnitsFeature(
        lambda x: FrequencyDecomposition(x, bands).as_frequency_adaptive(),
        needs=windowed)


def feature_map_size(inp, kernel, stride=1, padding=0):
    return ((inp - kernel + (2 * padding)) / stride) + 1


class FrequencyDecomposition(object):
    def __init__(self, samples, sizes, window=None):
        self.window = window
        self.sizes = sorted(sizes)
        self.samples = samples

        original = self.samples.copy()
        self.bands = []
        self.frequency_bands = []
        start_hz = 0

        for size in sizes:

            # extract a frequency band
            if size != self.size:
                s = self._resample(original, size)
            else:
                s = original

            self.bands.append(s)
            original -= self._resample(s, self.size)

            stop_hz = samplerate.nyquist * (size / self.size)
            self.frequency_bands.append(zounds.FrequencyBand(start_hz, stop_hz))
            start_hz = stop_hz

    @classmethod
    def _rs(cls, samples, desired_size, window=None):
        axis = -1
        w = window(samples.shape[axis]) if window else None
        return resample(samples, desired_size, axis=axis, window=w)

    def _resample(self, samples, desired_size):
        return self._rs(samples, desired_size, self.window)

    @classmethod
    def synthesize_block(cls, block, window=None):
        samples = np.zeros((len(block), SAMPLE_SIZE), dtype=block.dtype)
        start = 0
        for i, band in enumerate(bands):
            stop = start + band
            b = block[:, start: stop]
            samples += cls._rs(b * SCALING[i], SAMPLE_SIZE, window=window)
            start = stop
        return samples

    @property
    def size(self):
        return self.samples.shape[1]

    def as_frequency_adaptive(self):
        scale = zounds.ExplicitScale(self.frequency_bands)
        bands = [b / SCALING[i] for i, b in enumerate(self.bands)]
        return zounds.FrequencyAdaptive(
            bands, scale=scale, time_dimension=self.samples.dimensions[0])

    def synthesize_iter(self):
        fa = self.as_frequency_adaptive()
        samples = self.__class__.synthesize_block(fa)
        for sample in samples:
            yield sample, zounds.AudioSamples(sample, samplerate) \
                .pad_with_silence(zounds.Seconds(1))


class FDDiscriminator(nn.Module):
    def __init__(self):
        super(FDDiscriminator, self).__init__()
        self.factor = FACTOR
        self.layer_stacks = [[] for _ in bands]

        self.feature_map_sizes = reduce(
            lambda x, y: x + [int(x[-1] * FACTOR)],
            xrange(len(bands) - 1),
            [FIRST_FEATURE_MAP_SIZE])

        for i, band in enumerate(bands):
            for j in xrange(i + 1):
                fms = self.feature_map_sizes[j]
                first_layer = j == 0
                in_channels = 1 if first_layer else self.feature_map_sizes[
                    j - 1]
                params = (8, 4, 0) if first_layer else (3, 2, 0)
                layer = CriticLayer(in_channels, fms, *params)
                self.layer_stacks[i].append(layer)
                self.add_module('{i}{j}'.format(**locals()), layer)

        self.l1 = nn.Linear(sum(self.feature_map_sizes), 256, bias=False)
        self.l2 = nn.Linear(256, 1, bias=False)

    def forward(self, x):
        fms = []
        start_index = 0
        for i, band in enumerate(bands):
            stop = start_index + band
            slce = x[:, start_index: stop]
            start_index = stop
            fm = slce.contiguous().view(-1, 1, band)
            # subset = self.layers[:i + 1]
            subset = self.layer_stacks[i]
            for s in subset:
                fm = s(fm)
            fms.append(fm)

        # push the band-wise frequency maps through some linear layers
        flat = torch.cat(fms, dim=1).squeeze()
        x = self.l1(flat)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, 0.2, self.training)

        x = self.l2(x)
        return x


class FDGenerator(nn.Module):
    def __init__(self):
        super(FDGenerator, self).__init__()
        self.layer_stacks = [[] for _ in bands]
        self.factor = FACTOR
        self.feature_map_sizes = reduce(
            lambda x, y: x + [int(x[-1] * FACTOR)],
            xrange(len(bands) - 1),
            [FIRST_FEATURE_MAP_SIZE])

        for i, band in enumerate(bands):
            for j in xrange(i + 1):
                fms = self.feature_map_sizes[j]
                first_layer = j == 0
                out_channels = 1 if first_layer else self.feature_map_sizes[
                    j - 1]
                params = (8, 4, 0) if first_layer else (3, 2, 0)
                cls = FinalGeneratorLayer if first_layer else GeneratorLayer
                layer = cls(fms, out_channels, *params)
                self.layer_stacks[i].append(layer)
                self.add_module('{i}{j}'.format(**locals()), layer)

        total_features = sum(self.feature_map_sizes)
        self.l1 = nn.Linear(LATENT_DIM, 256, bias=False)
        self.bn1 = nn.BatchNorm1d(256)
        self.l2 = nn.Linear(256, total_features, bias=False)
        self.bn2 = nn.BatchNorm1d(total_features)

    def forward(self, x):

        x = x.view(-1, LATENT_DIM)
        x = self.l1(x)
        x = self.bn1(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, 0.2, self.training)

        x = self.l2(x)
        x = self.bn2(x)
        x = F.leaky_relu(x, 0.2)
        x = F.dropout(x, 0.2, self.training)

        current = 0
        bands = []
        for i, fms in enumerate(self.feature_map_sizes):
            stop = current + fms
            segment = x[:, current: stop]
            current = stop
            fm = segment.contiguous().view(-1, segment.size()[1], 1)
            # subset = self.layers[-(i + 1):]
            subset = self.layer_stacks[i][::-1]
            for s in subset:
                fm = s(fm)
            bands.append(fm.squeeze())

        return torch.cat(bands, dim=1)


class Generator(BaseGenerator):
    def __init__(self):
        super(Generator, self).__init__(
            (LATENT_DIM, 512, 4, 1, 0),
            (512, 256, 8, 4, 2),
            (256, 128, 8, 4, 2),
            (128, 128, 8, 4, 2),
            (128, 128, 8, 4, 2),
            (128, 1, 16, 8, 4))


class Generator2(BaseGenerator):
    def __init__(self):
        super(Generator2, self).__init__(
            (LATENT_DIM, 256, 4, 1, 0),
            (256, 256, 8, 4, 2),
            (256, 128, 8, 4, 2),
            (128, 128, 8, 4, 2),
            (128, 1, 126, 32, 47))


class Critic(BaseCritic):
    def __init__(self):
        super(Critic, self).__init__(
            SAMPLE_SIZE,
            (1, 64, 16, 8, 4),
            (64, 128, 8, 4, 2),
            (128, 128, 8, 4, 2),
            (128, 128, 8, 4, 2),
            (128, 256, 8, 4, 2),
            (256, 512, 4, 1, 0),
            (512, 1))


class Critic2(BaseCritic):
    def __init__(self):
        super(Critic2, self).__init__(
            SAMPLE_SIZE,
            (1, 128, 126, 32, 47),
            (128, 128, 8, 4, 2),
            (128, 256, 8, 4, 2),
            (256, 256, 8, 4, 2),
            (256, 512, 4, 1, 0),
            (512, 1))


class GanPair(nn.Module):
    def __init__(self):
        super(GanPair, self).__init__()
        self.generator = Generator()
        self.discriminator = Critic()

    def forward(self, x):
        raise NotImplementedError()


def try_network():
    z = np.random.normal(0, 1, (64, LATENT_DIM)).astype(np.float32)
    t = torch.from_numpy(z)
    v = Variable(t).cuda()
    network = GanPair().cuda()
    g = network.generator
    c = network.discriminator

    result = g(v, debug=True)
    print result.size()

    labels = c(result, debug=True)
    print labels.size()


@zounds.simple_settings
class Gan(ff.BaseModel):
    samples = ff.PickleFeature(ff.IteratorNode)

    shuffled = ff.PickleFeature(
        zounds.ShuffledSamples,
        nsamples=int(1e5),
        dtype=np.float32,
        needs=samples)

    scaled = ff.PickleFeature(
        zounds.InstanceScaling,
        needs=shuffled)

    wgan = ff.PickleFeature(
        zounds.PyTorchGan,
        trainer=ff.Var('trainer'),
        needs=scaled)

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(scaled, wgan,),
        store=True)


"""
Log
- reduced sample size from 8192 to 4096.  No difference
- Introduced batch norm.  Discriminator loss seems to stall out, and generator
  produces very periodic sine-like sound with many harmonics
- leaky RELU in discriminator seems to make little or no difference
- tanh for all layers produces noise
- leaky RELU in the generator seem to produce more plausible *looking* waveforms.
  generated examples are a combination of a single tone and noise
- add batch norm to the last generator layer - this seems to have really helped
   with the visual appearance of generated samples


- don't do instance scaling? there seems to be more variability, but still noise
- try with mu_law this results in noise, and strong peaks at convolution boundaries.  Why do things totally break down with mu_law?
- try the mu-law one-hot encoding.  this is very slow, and it's producing some pretty awful output.  Perhaps I should train longer with softmax.

- try penalizing the norm (improved WGAN) much more variation.  Still some noise
- try tanh all the way through - doesn't learn at all

- try interpolating in sample space - learns a good deal of variety
- try learning on downsampled 8192 samples -  there is some movement in the samples
- try without dropout - this seems to be slightly worse/noisier

- now that I'm doing WGAN, try instance scaling again - this seems to be OK now, and produces more variation
- try A-weighing the frequency domain and then IFFT back to samples- this seems to slightly improve variation
- try without tanh? - doesn't get rid of the noise
- try without batch norm in last layer? - doesn't change noise situation
- try with tanh in first layer of discriminator? - doesn't change noise


- can I scale up to 8192 and capture meaningful structure, even if noisy?
    - yes, at least for Bach
- try with phat drum loops
    - yes, starts to learn drum-like sounds
- try with speech
    - yes, starts to learn speech-like sounds
- try with a toy dataset so I can understand the biases/problems
- try with Kevin Gates
    - yes, learns speech-like sounds plus kick drums and bass
- try with a mixture of different types of sample

- residuals in the network
- dilated convolutions
- try adding batch norm back into discriminator?
- try training on mdct - nope, learns nothing
- progressively growing WGAN
"""


def load_and_play():
    files = sorted(
        glob.glob('*.npy'),
        cmp=lambda x, y: int(os.stat(x).st_ctime - os.stat(y).st_ctime))
    most_recent = files[-1]
    print 'loading generated examples from', most_recent
    results = np.load(most_recent)

    # synthesized = FrequencyDecomposition.synthesize_block(results)
    synthesized = results

    for raw, result in zip(results, synthesized):
        windowed = zounds.sliding_window(result, 512, 256)
        spec = np.abs(np.fft.rfft(windowed))
        audio_samples = zounds.AudioSamples(result, samplerate) \
            .pad_with_silence(zounds.Seconds(1))
        yield raw, result, audio_samples / audio_samples.max(), spec


def synthetic():
    for i in xrange(100):
        duration = zounds.Seconds(np.random.randint(2, 20))
        root = np.random.randint(50, 400)
        hz = [root]
        for _ in xrange(0):
            hz.append(hz[-1] * 2)
        synth = zounds.SineSynthesizer(samplerate)
        s = synth.synthesize(duration, hz)
        yield s.encode()


def ingest_all():
    data = [
        zounds.InternetArchive('AOC11B'),
        zounds.InternetArchive('Greatest_Speeches_of_the_20th_Century'),
        zounds.InternetArchive('Kevin_Gates_-_By_Any_Means-2014'),
        zounds.PhatDrumLoops()
    ]
    for d in data:
        zounds.ingest(d, Sound, multi_threaded=True)


def ingest():
    zounds.ingest(
        zounds.InternetArchive('AOC11B'),
        Sound,
        multi_threaded=True)

    # for s in synthetic():
    #     print Sound.process(meta=s, _id=uuid4().hex)


def ingest_and_train(epochs):
    ingest()

    network = GanPair()

    def arg_maker(epoch):
        z = np.random.normal(0, 1, (64, LATENT_DIM)).astype(np.float32)
        t = torch.from_numpy(z)
        v = Variable(t).cuda()
        samples = network.generator(v).data.cpu().numpy().squeeze()
        np.save('epoch' + str(epoch), samples)
        print 'saved samples for epoch', epoch
        return dict()

    # out_channels = 128
    # kernel_size = 126
    # basis = np.zeros((out_channels, kernel_size), dtype=np.float32)
    # synth = zounds.SineSynthesizer(samplerate)
    # space = np.geomspace(50, samplerate.nyquist, out_channels)
    # for i, freq in enumerate(space):
    #     basis[i] = synth.synthesize(samplerate.frequency * kernel_size, [freq])

    # basis = torch.from_numpy(basis[:, None, :]).cuda()
    # basis = Variable(basis)

    #
    # gen_basis = torch.from_numpy(basis[:, None, :]).cuda()
    # critic_basis = torch.from_numpy(basis[:, None, :]).cuda()
    #
    # print network.generator.main[-1].l1.weight.size()
    # network.generator.main[-1].l1.weight.data = gen_basis
    # network.generator.main[-1].l1.weight.requires_grad = False
    #
    # print network.discriminator.main[0].l1.weight.size()
    # network.discriminator.main[0].l1.weight.data = critic_basis
    # network.discriminator.main[0].l1.weight.requires_grad = False

    # def generator_loss_term(network, samples):
    #     result = F.conv1d(samples, basis, stride=64)
    #     result = torch.abs(result)
    #     mean = result.mean(dim=1)
    #     std = result.std(dim=1)
    #     result = mean / std
    #     return result.mean() * 2500

    if not Gan.exists():
        trainer = zounds.WassersteinGanTrainer(
            network=network,
            latent_dimension=(LATENT_DIM,),
            n_critic_iterations=10,
            epochs=epochs,
            batch_size=64,
            arg_maker=arg_maker)
        Gan.process(samples=(snd.perceptual for snd in Sound), trainer=trainer)

    p = Gan()

    def walk2(steps):
        for i in xrange(steps):
            yield np.random.normal(0, 1, LATENT_DIM)

    def listen():
        padding = zounds.Milliseconds(250)
        z = np.concatenate(list(walk2(1000)))
        result = p.pipeline.transform(z).data.squeeze()
        x = np.concatenate([
                               zounds.AudioSamples(j,
                                                   samplerate).pad_with_silence(
                                   padding)
                               for j in result])
        return zounds.AudioSamples(x, zounds.SR11025())

    return listen()


def test_frequency_decomposition():
    # snds = list(Sound)
    # snd = choice(snds)
    # fd = FrequencyDecomposition(snd.windowed, bands)
    # print [band.shape for band in fd.bands]

    gen = FDGenerator().cuda()

    inp = torch.zeros(64, LATENT_DIM).normal_(0, 1)
    inp = Variable(inp).cuda()
    x = gen(inp)

    print x.size()

    disc = FDDiscriminator().cuda()

    # fa = fd.as_frequency_adaptive()[:64].astype(np.float32)
    # print fa.shape, [fa[:, band].shape for band in fa.dimensions[1].scale]

    # inp = torch.from_numpy(fa)
    # inp = Variable(inp).cuda()

    x = disc(x)
    print x.size()

    print gen.layers
    print disc.layers


def tweak():
    snd = choice(list(Sound))
    original = snd.windowed
    windowed = original * np.hanning(SAMPLE_SIZE)
    twindowed = original * tukey(SAMPLE_SIZE, 0.2)

    ofd = FrequencyDecomposition(original, bands)
    ofd2 = FrequencyDecomposition(original, bands, window=np.hanning)

    wfd = FrequencyDecomposition(windowed, bands)
    wfd2 = FrequencyDecomposition(windowed, bands, window=np.hanning)

    tfd = FrequencyDecomposition(twindowed, bands)
    tfd2 = FrequencyDecomposition(twindowed, bands, window=np.hanning)
    return ofd, ofd2, wfd, wfd2, tfd, tfd2


def get_magnitudes():
    snds = list(Sound)
    snd = choice(snds)
    x = snd.decomposed

    magnitudes = []
    scaled = []

    start = 0
    for band in bands:
        stop = start + band
        b = x[:, start: stop]
        m = np.abs(b).mean()
        magnitudes.append(m)
        scaled.append(np.abs(b / m).mean())
    print magnitudes
    print scaled


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--ingest',
        action='store_true')
    parser.add_argument(
        '--train',
        help='ingest audio and train',
        action='store_true')
    parser.add_argument(
        '--epochs',
        help='number of epochs to train',
        type=int)
    parser.add_argument(
        '--try-network',
        help='dry run of network',
        action='store_true')
    parser.add_argument(
        '--evaluate',
        help='listen to and view generated results',
        action='store_true')
    parser.add_argument(
        '--test-decomposition',
        help='test out the frequency decomposition',
        action='store_true')
    parser.add_argument(
        '--get-magnitudes',
        action='store_true')
    parser.add_argument(
        '--tweak',
        action='store_true')

    args = parser.parse_args()
    if args.train:
        s = ingest_and_train(args.epochs)
    elif args.ingest:
        ingest()
    elif args.try_network:
        try_network()
    elif args.evaluate:
        result_iter = load_and_play()
    elif args.test_decomposition:
        test_frequency_decomposition()
    elif args.get_magnitudes:
        get_magnitudes()
    elif args.tweak:
        ofd, ofd2, wfd, wfd2, tfd, tfd2 = tweak()

    # start up an in-browser REPL to interact with the results
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.windowed,
        globals=globals(),
        locals=locals())
    app.start(9999)
