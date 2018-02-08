import featureflow as ff
from util import from_var
from random import choice
from zounds.spectral import stft, rainbowgram
from zounds.timeseries import SR11025, SampleRate, Seconds, AudioSamples
from wgan import WassersteinGanTrainer
from pytorch_model import PyTorchGan
from graph import learning_pipeline
from util import simple_settings
from preprocess import PreprocessingPipeline, InstanceScaling
from zounds.ui import ZoundsApp
from zounds.util import simple_lmdb_settings
from zounds.basic import resampled
from zounds.spectral import HanningWindowingFunc, SlidingWindow
from zounds.datasets import ingest
from zounds.persistence import ArrayWithUnitsFeature
import numpy as np


class GanExperiment(object):
    def __init__(
            self,
            experiment_name,
            dataset,
            gan_pair,
            epochs=500,
            n_critic_iterations=10,
            batch_size=32,
            n_samples=int(5e5),
            latent_dim=100,
            debug_gradients=False,
            sample_size=8192,
            sample_hop=1024,
            samplerate=SR11025(),
            app_port=8888):

        super(GanExperiment, self).__init__()
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

        base_model = resampled(
            resample_to=self.samplerate, store_resampled=True)

        window_sample_rate = SampleRate(
            frequency=self.samplerate.frequency * sample_hop,
            duration=samplerate.frequency * sample_size)

        @simple_lmdb_settings(
            experiment_name, map_size=1e11, user_supplied_id=True)
        class Sound(base_model):
            windowed = ArrayWithUnitsFeature(
                SlidingWindow,
                wscheme=window_sample_rate,
                needs=base_model.resampled)

        self.sound_cls = Sound

        base_pipeline = learning_pipeline()

        @simple_settings
        class Gan(base_pipeline):
            scaled = ff.PickleFeature(
                InstanceScaling,
                needs=base_pipeline.shuffled)

            wgan = ff.PickleFeature(
                PyTorchGan,
                trainer=ff.Var('trainer'),
                needs=scaled)

            pipeline = ff.PickleFeature(
                PreprocessingPipeline,
                needs=(scaled, wgan),
                store=True)

        self.gan_pipeline = Gan()
        self.fake_samples = None
        self.app = None

    def batch_complete(self, epoch, network, samples):
        self.fake_samples = from_var(samples).squeeze()

    def fake_audio(self):
        sample = choice(self.fake_samples)
        return AudioSamples(sample, self.samplerate)\
            .pad_with_silence(Seconds(1))

    def fake_stft(self):
        samples = self.fake_audio()
        wscheme = SampleRate(
            frequency=samples.samplerate.frequency * 128,
            duration=samples.samplerate.frequency * 256)
        coeffs = stft(samples, wscheme, HanningWindowingFunc())
        return rainbowgram(coeffs)

    def run(self):
        ingest(self.dataset, self.sound_cls, multi_threaded=True)

        experiment = self
        fake_audio = self.fake_audio
        fake_stft = self.fake_stft

        self.app = ZoundsApp(
            model=self.sound_cls,
            audio_feature=self.sound_cls.ogg,
            visualization_feature=self.sound_cls.windowed,
            globals=globals(),
            locals=locals())

        with self.app.start_in_thread(self.app_port):
            if not self.gan_pipeline.exists():
                network = self.gan_pair

                for p in network.parameters():
                    p.data.normal_(0, 0.02)

                trainer = WassersteinGanTrainer(
                    network,
                    latent_dimension=(self.latent_dim,),
                    n_critic_iterations=self.n_critic_iterations,
                    epochs=self.epochs,
                    batch_size=self.batch_size,
                    on_batch_complete=self.batch_complete,
                    debug_gradient=self.debug_gradients)

                self.gan_pipeline.process(
                    samples=(snd.windowed for snd in self.sound_cls),
                    trainer=trainer,
                    nsamples=self.n_samples,
                    dtype=np.float32)

        self.app.start(self.app_port)
