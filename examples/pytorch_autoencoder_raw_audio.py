import argparse
from random import choice

import featureflow as ff
import numpy as np
from torch import nn
from torch.nn import functional as F
from torch.optim import Adam

import zounds
from zounds.learn import Conv1d, ConvTranspose1d, to_var, from_var
from zounds.timeseries import categorical, inverse_categorical

samplerate = zounds.SR11025()
BaseModel = zounds.resampled(resample_to=samplerate, store_resampled=True)

window_size = 8192
wscheme = zounds.SampleRate(
    frequency=samplerate.frequency * (window_size // 2),
    duration=samplerate.frequency * window_size)


@zounds.simple_lmdb_settings('ae', map_size=1e10, user_supplied_id=True)
class Sound(BaseModel):
    windowed = zounds.ArrayWithUnitsFeature(
        zounds.SlidingWindow,
        wscheme=wscheme,
        needs=BaseModel.resampled)

    mu_law = zounds.ArrayWithUnitsFeature(
        zounds.mu_law,
        needs=windowed)

    categorical = zounds.ArrayWithUnitsFeature(
        categorical,
        needs=windowed)


# TODO: Factor out the part of the pipeline that starts with samples and
# shuffled
@zounds.simple_settings
class AutoEncoderPipeline(ff.BaseModel):
    samples = ff.PickleFeature(ff.IteratorNode)

    shuffled = ff.PickleFeature(
        zounds.ShuffledSamples,
        nsamples=int(1e5),
        dtype=np.float32,
        needs=samples)

    scaled = ff.PickleFeature(
        zounds.InstanceScaling,
        needs=shuffled)

    autoencoder = ff.PickleFeature(
        zounds.PyTorchAutoEncoder,
        trainer=ff.Var('trainer'),
        needs=scaled)

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(scaled, autoencoder,),
        store=True)


@zounds.simple_settings
class CategoricalAutoEncoderPipeline(ff.BaseModel):
    samples = ff.PickleFeature(ff.IteratorNode)

    shuffled = ff.PickleFeature(
        zounds.ShuffledSamples,
        nsamples=int(1e5),
        dtype=np.float32,
        needs=samples)

    autoencoder = ff.PickleFeature(
        zounds.PyTorchAutoEncoder,
        trainer=ff.Var('trainer'),
        needs=shuffled)

    pipeline = ff.PickleFeature(
        zounds.PreprocessingPipeline,
        needs=(autoencoder,),
        store=True)


class EncoderLayer(Conv1d):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding):
        super(EncoderLayer, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding)


class DecoderLayer(ConvTranspose1d):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout=True,
            activation=lambda x: F.leaky_relu(x, 0.2)):
        super(DecoderLayer, self).__init__(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation,
            dropout)


class Encoder(nn.Module):
    def __init__(self, in_channels):
        super(Encoder, self).__init__()
        self.in_channels = in_channels
        self.main = nn.Sequential(
            EncoderLayer(in_channels, 64, 16, 8, 4),
            EncoderLayer(64, 128, 8, 4, 2),
            EncoderLayer(128, 128, 8, 4, 2),
            EncoderLayer(128, 128, 8, 4, 2),
            EncoderLayer(128, 256, 8, 4, 2),
            EncoderLayer(256, 512, 4, 1, 0))

    def forward(self, x):
        x = x.view(-1, self.in_channels, window_size)
        return self.main(x).view(-1, 512)


class Decoder(nn.Module):
    def __init__(self, out_channels, output_activation):
        super(Decoder, self).__init__()
        act = output_activation
        self.out_channels = out_channels
        self.main = nn.Sequential(
            DecoderLayer(512, 256, 4, 1, 0),
            DecoderLayer(256, 128, 8, 4, 2),
            DecoderLayer(128, 128, 8, 4, 2),
            DecoderLayer(128, 128, 8, 4, 2),
            DecoderLayer(128, 64, 8, 4, 2),
            DecoderLayer(
                64, self.out_channels, 16, 8, 4, dropout=False, activation=act))

    def forward(self, x):
        x = x.view(-1, 512, 1)
        x = self.main(x)
        x = x.view(-1, self.out_channels, window_size)
        x = x.squeeze()
        return x


class AutoEncoder(nn.Module):
    def __init__(self, channels, output_activation):
        super(AutoEncoder, self).__init__()
        self.encoder = Encoder(channels)
        self.decoder = Decoder(channels, output_activation)

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


class RawSamplesAutoEncoder(AutoEncoder):
    def __init__(self):
        super(RawSamplesAutoEncoder, self).__init__(
            channels=1, output_activation=F.tanh)


class CategoricalAutoEncoder(AutoEncoder):
    def __init__(self):
        super(CategoricalAutoEncoder, self).__init__(
            channels=256, output_activation=F.log_softmax)


def raw_samples_synthesize(x):
    # TODO: it should be possible to apply windowing at the synthesis step
    synth = zounds.WindowedAudioSynthesizer()
    return synth.synthesize(x)


def categorical_synthesize(x):
    samples = inverse_categorical(x.reshape(-1, 8192, 256))
    samples = zounds.ArrayWithUnits(samples, dimensions=[
        zounds.TimeDimension(*wscheme),
        zounds.TimeDimension(*samplerate)
    ])
    return raw_samples_synthesize(samples)


def preprocess_categorical(x):
    return categorical(x).reshape((-1, 256, 8192))


class CategoricalLoss(nn.NLLLoss):
    def __init__(self):
        super(CategoricalLoss, self).__init__()

    def forward(self, input, target):
        input = input.view(-1, 256)
        target = target.view(-1, 256)
        values, indices = target.max(dim=1)
        return super(CategoricalLoss, self).forward(input, indices)


class FrequencyBandLoss(nn.MSELoss):
    def __init__(self):
        super(FrequencyBandLoss, self).__init__()

    def forward(self, input, target):
        target_samples = from_var(target).squeeze()
        target_fft = np.fft.rfft(target_samples, axis=-1, norm='ortho')
        target_fft[:, :50] = 0
        recon = np.fft.irfft(target_fft, axis=-1, norm='ortho')
        recon = to_var(recon)
        return super(FrequencyBandLoss, self).forward(input, recon)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--internet-archive-id',
        type=str,
        help='the internet archive id to use for training')
    parser.add_argument(
        '--epochs',
        type=int,
        help='the number of epochs to train the network')
    parser.add_argument(
        '--force-train',
        action='store_true',
        help='re-train the network, even if it has already been trained')
    parser.add_argument(
        '--categorical',
        action='store_true',
        help='use a categorical distribution of samples')
    args = parser.parse_args()

    if args.internet_archive_id:
        zounds.ingest(
            zounds.InternetArchive(args.internet_archive_id),
            Sound,
            multi_threaded=True)

    if args.categorical:
        network = CategoricalAutoEncoder()
        loss = CategoricalLoss()
        synthesize = categorical_synthesize
        pipeline_cls = CategoricalAutoEncoderPipeline
        data_preprocessor = label_preprocessor = preprocess_categorical
        batch_size = 16
    else:
        network = RawSamplesAutoEncoder()
        loss = FrequencyBandLoss()
        synthesize = raw_samples_synthesize
        pipeline_cls = AutoEncoderPipeline
        data_preprocessor = label_preprocessor = lambda x: x
        batch_size = 64
        gen = (snd.windowed for snd in Sound
               if args.internet_archive_id in snd._id)

    if args.force_train or not AutoEncoderPipeline.exists():
        trainer = zounds.SupervisedTrainer(
            network,
            loss,
            lambda model: Adam(model.parameters(), lr=0.0001),
            epochs=args.epochs,
            batch_size=batch_size,
            holdout_percent=0.25,
            data_preprocessor=data_preprocessor,
            label_preprocessor=label_preprocessor)

        gen = (snd.windowed for snd in Sound
               if args.internet_archive_id in snd._id)
        pipeline_cls.process(samples=gen, trainer=trainer)

    # instantiate the trained pipeline
    pipeline = pipeline_cls()

    snds = filter(lambda snd: args.internet_archive_id in snd._id, Sound)
    snd = choice(snds)
    time_slice = zounds.TimeSlice(duration=zounds.Seconds(10))
    encoded = pipeline.pipeline.transform(
        data_preprocessor(snd.windowed[time_slice]))
    recon = encoded.inverse_transform()
    samples = synthesize(recon)

    # start up an in-browser REPL to interact with the results
    app = zounds.ZoundsApp(
        model=Sound,
        audio_feature=Sound.ogg,
        visualization_feature=Sound.windowed,
        globals=globals(),
        locals=locals())
    app.start(8888)
