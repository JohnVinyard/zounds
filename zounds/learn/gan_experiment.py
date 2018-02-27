import featureflow as ff
from util import from_var
from random import choice
from zounds.spectral import stft, rainbowgram
from zounds.learn import \
    try_network, infinite_streaming_learning_pipeline, \
    object_store_pipeline_settings
from zounds.timeseries import \
    SR11025, SampleRate, Seconds, AudioSamples, audio_sample_rate
from wgan import WassersteinGanTrainer
from pytorch_model import PyTorchGan
from preprocess import InstanceScaling
from zounds.ui import GanTrainingMonitorApp
from zounds.util import simple_lmdb_settings
from zounds.basic import windowed
from zounds.spectral import HanningWindowingFunc
from zounds.datasets import ingest
import numpy as np


class GanExperiment(object):
    def __init__(
            self,
            experiment_name,
            dataset,
            gan_pair,
            object_storage_username,
            object_storage_api_key,
            epochs=500,
            n_critic_iterations=10,
            batch_size=32,
            n_samples=int(5e5),
            latent_dim=100,
            real_sample_transformer=lambda x: x,
            debug_gradients=False,
            sample_size=8192,
            sample_hop=1024,
            samplerate=SR11025(),
            app_port=8888,
            object_storage_region='DFW',
            app_secret=None):

        super(GanExperiment, self).__init__()
        self.real_sample_transformer = real_sample_transformer
        self.debug_gradients = debug_gradients
        self.n_samples = n_samples
        self.batch_size = batch_size
        self.n_critic_iterations = n_critic_iterations
        self.epochs = epochs
        self.gan_pair = gan_pair
        self.app_port = app_port
        self.dataset = dataset

        self.samplerate = samplerate
        self.sample_hop = sample_hop
        self.sample_size = sample_size
        self.latent_dim = latent_dim
        self.experiment_name = experiment_name
        self.app_secret = app_secret

        base_model = windowed(
            resample_to=self.samplerate,
            store_resampled=True,
            wscheme=self.samplerate * (sample_hop, sample_size))

        @simple_lmdb_settings(
            experiment_name, map_size=1e11, user_supplied_id=True)
        class Sound(base_model):
            pass

        self.sound_cls = Sound

        @object_store_pipeline_settings(
            'Gan-{experiment_name}'.format(**locals()),
            object_storage_region,
            object_storage_username,
            object_storage_api_key)
        @infinite_streaming_learning_pipeline
        class Gan(ff.BaseModel):
            scaled = ff.PickleFeature(
                InstanceScaling)

            wgan = ff.PickleFeature(
                PyTorchGan,
                trainer=ff.Var('trainer'),
                needs=scaled)

        self.gan_pipeline = Gan()
        self.fake_samples = None
        self.app = None

    def batch_complete(self, *args, **kwargs):
        samples = kwargs['samples']
        self.fake_samples = from_var(samples).squeeze()

    def fake_audio(self):
        sample = choice(self.fake_samples)
        return AudioSamples(sample, self.samplerate) \
            .pad_with_silence(Seconds(1))

    def _stft(self, samples):
        samples = samples / np.abs(samples.max())
        wscheme = SampleRate(
            frequency=samples.samplerate.frequency * 128,
            duration=samples.samplerate.frequency * 256)
        coeffs = stft(samples, wscheme, HanningWindowingFunc())
        return rainbowgram(coeffs)

    def fake_stft(self):
        samples = self.fake_audio()
        return self._stft(samples)

    def real_stft(self):
        snd = self.sound_cls.random()
        windowed = choice(snd.windowed)
        windowed = AudioSamples(
            windowed,
            audio_sample_rate(windowed.dimensions[0].samples_per_second))
        return self._stft(windowed)

    def test(self):
        z = np.random.normal(
            0, 1, (self.batch_size, self.latent_dim)).astype(np.float32)
        samples = try_network(self.gan_pair.generator, z)
        samples = from_var(samples)
        print samples.shape
        wasserstein_estimate = try_network(self.gan_pair.discriminator, samples)
        print wasserstein_estimate.shape

    def run(self):
        ingest(self.dataset, self.sound_cls, multi_threaded=True)

        experiment = self
        fake_audio = self.fake_audio
        fake_stft = self.fake_stft
        real_stft = self.real_stft
        Sound = self.sound_cls

        try:
            network = self.gan_pipeline.load_network()
            print 'initialized weights'
        except RuntimeError as e:
            print 'Error', e
            network = self.gan_pair
            for p in network.parameters():
                p.data.normal_(0, 0.02)

        trainer = WassersteinGanTrainer(
            network,
            latent_dimension=(self.latent_dim,),
            n_critic_iterations=self.n_critic_iterations,
            epochs=self.epochs,
            batch_size=self.batch_size,
            debug_gradient=self.debug_gradients)
        trainer.register_batch_complete_callback(self.batch_complete)

        self.app = GanTrainingMonitorApp(
            trainer=trainer,
            model=Sound,
            visualization_feature=Sound.windowed,
            audio_feature=Sound.ogg,
            globals=globals(),
            locals=locals(),
            secret=self.app_secret)

        with self.app.start_in_thread(self.app_port):
            self.gan_pipeline.process(
                dataset=(Sound, Sound.windowed),
                trainer=trainer,
                nsamples=self.n_samples,
                dtype=np.float32)

        self.app.start(self.app_port)
