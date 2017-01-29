import unittest2
from random_samples import ReservoirSampler
from zounds.timeseries import TimeDimension, Seconds
from zounds.spectral import FrequencyDimension, FrequencyBand, LinearScale
from zounds.core import ArrayWithUnits, IdentityDimension
import numpy as np


class TestReservoirSampler(unittest2.TestCase):

    def test_can_wrap_samples(self):
        def wrapper(empty, orig):
            return ArrayWithUnits(
                    empty, [IdentityDimension()] + list(orig.dimensions[1:]))

        sampler = ReservoirSampler(
                nsamples=10,
                wrapper=wrapper)

        frequency_dimension = FrequencyDimension(
                LinearScale(FrequencyBand(100, 1000), 100))

        samples = ArrayWithUnits(
                np.ones((2, 10, 100)),
                [
                    TimeDimension(frequency=Seconds(10)),
                    TimeDimension(frequency=Seconds(1)),
                    frequency_dimension
                ])

        sampler._enqueue(samples, pusher=None)
        reservoir = sampler._r
        self.assertEqual((10, 10, 100), reservoir.shape)
        self.assertIsInstance(reservoir, ArrayWithUnits)
        self.assertEqual(reservoir.dimensions[0], IdentityDimension())
        self.assertEqual(reservoir.dimensions[1], samples.dimensions[1])
        self.assertEqual(reservoir.dimensions[2], samples.dimensions[2])

    def test_default_wrapper_is_identity_function(self):

        sampler = ReservoirSampler(nsamples=10)

        frequency_dimension = FrequencyDimension(
                LinearScale(FrequencyBand(100, 1000), 100))

        samples = ArrayWithUnits(
                np.ones((2, 10, 100)),
                [
                    TimeDimension(frequency=Seconds(10)),
                    TimeDimension(frequency=Seconds(1)),
                    frequency_dimension
                ])

        sampler._enqueue(samples, pusher=None)
        reservoir = sampler._r
        self.assertEqual((10, 10, 100), reservoir.shape)
        self.assertNotIsInstance(reservoir, ArrayWithUnits)
        self.assertIsInstance(reservoir, np.ndarray)
