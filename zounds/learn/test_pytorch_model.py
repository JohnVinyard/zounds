from __future__ import division

import featureflow as ff
import numpy as np
import unittest2

from preprocess import \
    UnitNorm, Binarize, PreprocessingPipeline, InstanceScaling
from pytorch_model import PyTorchNetwork, PyTorchAutoEncoder, PyTorchGan
from supervised import SupervisedTrainer
from wgan import WassersteinGanTrainer
from random_samples import ShuffledSamples
from zounds.core import ArrayWithUnits, IdentityDimension
from zounds.spectral import LinearScale, FrequencyBand, FrequencyDimension
from zounds.timeseries import Seconds, TimeDimension
from zounds.util import simple_in_memory_settings

try:
    import torch
    from torch import nn
    from torch.optim import SGD, Adam


    class SupervisedNetwork(nn.Module):
        def __init__(self):
            super(SupervisedNetwork, self).__init__()
            self.visible = nn.Linear(2, 64, bias=False)
            self.t1 = nn.Sigmoid()
            self.hidden = nn.Linear(64, 1, bias=False)
            self.t2 = nn.Sigmoid()

        def forward(self, inp):
            x = self.visible(inp)
            x = self.t1(x)
            x = self.hidden(x)
            x = self.t2(x)
            return x


    class AutoEncoder(nn.Module):
        def __init__(self):
            super(AutoEncoder, self).__init__()

            self.encoder = nn.Sequential(
                nn.Linear(3, 2, bias=False),
                nn.Sigmoid())

            self.decoder = nn.Sequential(
                nn.Linear(2, 3, bias=False),
                nn.Sigmoid())

        def forward(self, inp):
            x = self.encoder(inp)
            x = self.decoder(x)
            return x


    class GanGenerator(nn.Module):
        def __init__(self):
            super(GanGenerator, self).__init__()
            self.linear = nn.Linear(2, 4)
            self.tanh = nn.Tanh()

        def forward(self, inp):
            x = self.linear(inp)
            x = self.tanh(x)
            return x


    class GanDiscriminator(nn.Module):
        def __init__(self):
            super(GanDiscriminator, self).__init__()
            self.linear = nn.Linear(4, 1)
            self.sigmoid = nn.Sigmoid()

        def forward(self, inp):
            x = self.linear(inp)
            x = self.sigmoid(x)
            return x


    class GanPair(nn.Module):
        def __init__(self):
            super(GanPair, self).__init__()
            self.generator = GanGenerator()
            self.discriminator = GanDiscriminator()

        def forward(self, x):
            raise NotImplementedError()

except ImportError:
    torch = None


class PyTorchModelTests(unittest2.TestCase):
    def setUp(self):
        if torch is None:
            self.skipTest('pytorch is not available')

    def test_can_maintain_array_dimensions_with_supervised_learning(self):
        trainer = SupervisedTrainer(
            model=SupervisedNetwork(),
            loss=nn.BCELoss(),
            optimizer=lambda model: SGD(model.parameters(), lr=0.2),
            epochs=1,
            batch_size=64)

        @simple_in_memory_settings
        class Pipeline(ff.BaseModel):
            inp = ff.PickleFeature(
                ff.IteratorNode,
                store=False)

            samples = ff.PickleFeature(
                ShuffledSamples,
                nsamples=500,
                multiplexed=True,
                needs=inp,
                store=False)

            unitnorm = ff.PickleFeature(
                UnitNorm,
                needs=samples.aspect('data'),
                store=False)

            hard_labels = ff.PickleFeature(
                Binarize,
                needs=samples.aspect('labels'),
                store=False)

            network = ff.PickleFeature(
                PyTorchNetwork,
                trainer=trainer,
                needs=dict(data=unitnorm, labels=hard_labels),
                store=False)

            pipeline = ff.PickleFeature(
                PreprocessingPipeline,
                needs=(unitnorm, network),
                store=True)

        # Produce some random points on the unit circle
        samples = np.random.random_sample((1000, 2))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)

        # a line extending from the origin to (1, 1)
        origin = np.array([0, 0])
        unit = np.array([1, 1])

        # which side of the plane is each sample on?
        labels = np.sign(np.cross(unit - origin, origin - samples))
        labels[labels < 0] = 0

        # scale each sample randomly, forcing the pipeline to normalize data
        factors = np.random.randint(1, 1000, (len(samples), 1))
        scaled_samples = samples * factors
        scaled_samples = scaled_samples

        # fuzz the labels, forcing the pipeline to binarize these (i.e., force
        # them to be 0 or 1)
        fuzzed_labels = labels + np.random.normal(0, 0.1, labels.shape)
        fuzzed_labels = fuzzed_labels[..., None]

        def gen(chunksize, s, l):
            for i in xrange(0, len(s), chunksize):
                sl = slice(i, i + chunksize)
                yield dict(data=s[sl], labels=l[sl])

        _id = Pipeline.process(inp=gen(100, scaled_samples, fuzzed_labels))
        pipe = Pipeline(_id)

        # produce some new samples
        new_samples = np.random.random_sample((1000, 2))
        new_samples /= np.linalg.norm(samples, axis=1, keepdims=True)

        # scale each example randomly, so the pipeline must give it unit norm
        # to arrive at the correct answer
        new_factors = np.random.randint(1, 1000, (len(samples), 1))
        new_scaled_samples = new_factors * new_samples

        arr = ArrayWithUnits(
            new_scaled_samples,
            dimensions=[
                TimeDimension(Seconds(1)),
                FrequencyDimension(LinearScale(FrequencyBand(100, 1000), 2))
            ])

        result = pipe.pipeline.transform(arr)
        self.assertIsInstance(result.data, ArrayWithUnits)
        self.assertIsInstance(result.data.dimensions[0], TimeDimension)

    def test_can_perform_supervised_learning(self):
        """
        Create and exercise a learning pipeline that learn to classify
        2d points as being on one side or the other of a plane from (0, 0) to
        (1, 1)
        """

        trainer = SupervisedTrainer(
            model=SupervisedNetwork(),
            loss=nn.BCELoss(),
            optimizer=lambda model: SGD(model.parameters(), lr=0.2),
            epochs=100,
            batch_size=64)

        @simple_in_memory_settings
        class Pipeline(ff.BaseModel):
            inp = ff.PickleFeature(
                ff.IteratorNode,
                store=False)

            samples = ff.PickleFeature(
                ShuffledSamples,
                nsamples=500,
                multiplexed=True,
                needs=inp,
                store=False)

            unitnorm = ff.PickleFeature(
                UnitNorm,
                needs=samples.aspect('data'),
                store=False)

            hard_labels = ff.PickleFeature(
                Binarize,
                needs=samples.aspect('labels'),
                store=False)

            network = ff.PickleFeature(
                PyTorchNetwork,
                trainer=trainer,
                needs=dict(data=unitnorm, labels=hard_labels),
                store=False)

            pipeline = ff.PickleFeature(
                PreprocessingPipeline,
                needs=(unitnorm, network),
                store=True)

        # Produce some random points on the unit circle
        samples = np.random.random_sample((1000, 2))
        samples /= np.linalg.norm(samples, axis=1, keepdims=True)

        # a line extending from the origin to (1, 1)
        origin = np.array([0, 0])
        unit = np.array([1, 1])

        # which side of the plane is each sample on?
        labels = np.sign(np.cross(unit - origin, origin - samples))
        labels[labels < 0] = 0

        # scale each sample randomly, forcing the pipeline to normalize data
        factors = np.random.randint(1, 1000, (len(samples), 1))
        scaled_samples = samples * factors
        scaled_samples = scaled_samples

        # fuzz the labels, forcing the pipeline to binarize these (i.e., force
        # them to be 0 or 1)
        fuzzed_labels = labels + np.random.normal(0, 0.1, labels.shape)
        fuzzed_labels = fuzzed_labels[..., None]

        def gen(chunksize, s, l):
            for i in xrange(0, len(s), chunksize):
                sl = slice(i, i + chunksize)
                yield dict(data=s[sl], labels=l[sl])

        _id = Pipeline.process(inp=gen(100, scaled_samples, fuzzed_labels))
        pipe = Pipeline(_id)

        # produce some new samples
        new_samples = np.random.random_sample((1000, 2))
        new_samples /= np.linalg.norm(samples, axis=1, keepdims=True)

        # which side of the plane is each sample on?
        new_labels = np.sign(np.cross(unit - origin, origin - new_samples))
        new_labels[new_labels < 0] = 0

        # scale each example randomly, so the pipeline must give it unit norm
        # to arrive at the correct answer
        new_factors = np.random.randint(1, 1000, (len(samples), 1))
        new_scaled_samples = new_factors * new_samples

        result = pipe.pipeline.transform(new_scaled_samples)

        # reshape the data, and normalize to 0 or 1
        result = np.round(result.data.squeeze())

        # compute the number of labels that are incorrect
        difference = np.logical_xor(result, new_labels).sum()
        percent_error = difference / len(result)

        self.assertLess(percent_error, 0.05)

    def test_maintains_array_with_units_dimensions(self):
        trainer = SupervisedTrainer(
            AutoEncoder(),
            loss=nn.MSELoss(),
            optimizer=lambda model: SGD(model.parameters(), lr=0.1),
            epochs=2,
            batch_size=64)

        @simple_in_memory_settings
        class Pipeline(ff.BaseModel):
            inp = ff.PickleFeature(
                ff.IteratorNode,
                store=False)

            samples = ff.PickleFeature(
                ShuffledSamples,
                nsamples=500,
                needs=inp,
                store=False)

            unitnorm = ff.PickleFeature(
                UnitNorm,
                needs=samples,
                store=False)

            network = ff.PickleFeature(
                PyTorchAutoEncoder,
                trainer=trainer,
                needs=unitnorm,
                store=False)

            pipeline = ff.PickleFeature(
                PreprocessingPipeline,
                needs=(unitnorm, network),
                store=True)

        training = np.random.random_sample((1000, 3))

        def gen(chunksize, s):
            for i in xrange(0, len(s), chunksize):
                yield s[i: i + chunksize]

        _id = Pipeline.process(inp=gen(100, training))
        pipe = Pipeline(_id)

        test = ArrayWithUnits(
            np.random.random_sample((10, 3)),
            dimensions=[
                TimeDimension(Seconds(1)),
                FrequencyDimension(LinearScale(FrequencyBand(100, 1000), 3))
            ])
        result = pipe.pipeline.transform(test)
        self.assertEqual((10, 2), result.data.shape)
        self.assertIsInstance(result.data, ArrayWithUnits)
        self.assertIsInstance(result.data.dimensions[0], TimeDimension)
        self.assertIsInstance(result.data.dimensions[1], IdentityDimension)

        inverted = result.inverse_transform()
        self.assertEqual((10, 3), inverted.shape)
        self.assertIsInstance(inverted, ArrayWithUnits)
        self.assertIsInstance(inverted.dimensions[0], TimeDimension)
        self.assertIsInstance(inverted.dimensions[1], FrequencyDimension)

    def test_can_perform_unsupervised_learning_autoencoder(self):

        trainer = SupervisedTrainer(
            AutoEncoder(),
            loss=nn.MSELoss(),
            optimizer=lambda model: SGD(model.parameters(), lr=0.1),
            epochs=10,
            batch_size=64)

        @simple_in_memory_settings
        class Pipeline(ff.BaseModel):
            inp = ff.PickleFeature(
                ff.IteratorNode,
                store=False)

            samples = ff.PickleFeature(
                ShuffledSamples,
                nsamples=500,
                needs=inp,
                store=False)

            unitnorm = ff.PickleFeature(
                UnitNorm,
                needs=samples,
                store=False)

            network = ff.PickleFeature(
                PyTorchAutoEncoder,
                trainer=trainer,
                needs=unitnorm,
                store=False)

            pipeline = ff.PickleFeature(
                PreprocessingPipeline,
                needs=(unitnorm, network),
                store=True)

        training = np.random.random_sample((1000, 3))

        def gen(chunksize, s):
            for i in xrange(0, len(s), chunksize):
                yield s[i: i + chunksize]

        _id = Pipeline.process(inp=gen(100, training))
        pipe = Pipeline(_id)

        test = np.random.random_sample((10, 3))
        result = pipe.pipeline.transform(test)
        self.assertEqual((10, 2), result.data.shape)

        inverted = result.inverse_transform()
        self.assertEqual((10, 3), inverted.shape)

    def test_can_train_gan(self):

        trainer = WassersteinGanTrainer(
            GanPair(),
            latent_dimension=(2,),
            n_critic_iterations=5,
            epochs=10,
            batch_size=64)

        @simple_in_memory_settings
        class Pipeline(ff.BaseModel):
            inp = ff.PickleFeature(
                ff.IteratorNode,
                store=False)

            samples = ff.PickleFeature(
                ShuffledSamples,
                nsamples=500,
                needs=inp,
                store=False)

            scaled = ff.PickleFeature(
                InstanceScaling,
                needs=samples,
                store=False)

            network = ff.PickleFeature(
                PyTorchGan,
                apply_network='generator',
                trainer=trainer,
                needs=scaled,
                store=False)

            pipeline = ff.PickleFeature(
                PreprocessingPipeline,
                needs=(scaled, network),
                store=True)

        training_data = np.random.normal(0, 1, (1000, 4))

        def gen(chunksize, s):
            for i in xrange(0, len(s), chunksize):
                yield s[i: i + chunksize]

        _id = Pipeline.process(inp=gen(100, training_data))
        pipe = Pipeline(_id)

        noise = np.random.normal(0, 1, (10, 2))
        result = pipe.pipeline.transform(noise)
        self.assertEqual((10, 4), result.data.shape)
