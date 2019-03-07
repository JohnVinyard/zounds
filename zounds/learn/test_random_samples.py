import unittest2
from .random_samples import \
    ReservoirSampler, Reservoir, MultiplexedReservoir
from zounds.timeseries import TimeDimension, Seconds
from zounds.spectral import FrequencyDimension, FrequencyBand, LinearScale
from zounds.core import ArrayWithUnits, IdentityDimension
import numpy as np


class TestReservoir(unittest2.TestCase):
    def test_nsamples_must_be_gt_zero(self):
        self.assertRaises(ValueError, lambda: Reservoir(0))

    def test_can_dictate_dtype(self):
        r = Reservoir(100, dtype=np.float32)
        r.add(np.ones(10, dtype=np.float64))
        self.assertEqual(np.float32, r.get().dtype)

    def test_reservoir_has_first_input_dtype_when_unspecified(self):
        r = Reservoir(100)
        r.add(np.ones(10, dtype=np.float64))
        self.assertEqual(np.float64, r.get().dtype)

    def test_raises_if_nsamples_is_not_int(self):
        self.assertRaises(ValueError, lambda: Reservoir(1e2))

    def test_array_has_correct_first_dimension(self):
        r = Reservoir(100)
        r.add(np.random.random_sample((10, 3)))
        self.assertEqual(100, r.arr.shape[0])

    def test_can_add_samples_larger_than_reservoir_size(self):
        r = Reservoir(100)
        r.add(np.random.random_sample((1000, 3)))
        self.assertEqual(100, len(r.get()))

    def test_array_has_correct_subsequent_dimensions(self):
        r = Reservoir(100)
        r.add(np.random.random_sample((10, 3, 2)))
        self.assertEqual((3, 2), r.arr.shape[1:])

    def test_array_with_units(self):
        r = Reservoir(100)

        frequency_dimension = FrequencyDimension(
            LinearScale(FrequencyBand(100, 1000), 100))

        samples = ArrayWithUnits(
            np.ones((20, 100)),
            [
                TimeDimension(frequency=Seconds(1)),
                frequency_dimension
            ])

        r.add(samples)
        mixed = r.get()
        self.assertIsInstance(mixed, ArrayWithUnits)
        self.assertEqual(100, mixed.shape[1])
        self.assertIsInstance(mixed.dimensions[0], IdentityDimension)
        self.assertIsInstance(mixed.dimensions[1], FrequencyDimension)

    def test_reservoir_is_well_mixed(self):
        r = Reservoir(100)
        samples = np.arange(100)[..., None]
        for i in range(0, 100, 10):
            r.add(samples[i: i + 10])
        mixed = r.get().squeeze()
        diff = np.diff(mixed)
        self.assertFalse(np.all(diff == 1))

    def test_can_provide_explicit_indices_when_adding(self):
        r = Reservoir(10)
        samples = np.arange(10)[..., None]
        r.add(samples, indices=samples.squeeze()[::-1])
        mixed = r.get()
        np.testing.assert_allclose(mixed.squeeze(), samples.squeeze()[::-1])

    def test_raises_when_samples_and_explicit_indices_dont_match(self):
        r = Reservoir(10)
        samples = np.arange(10)[..., None]
        self.assertRaises(
            ValueError, lambda: r.add(samples, indices=samples.squeeze()[:5]))

    def test_can_get_batch(self):
        r = Reservoir(100)
        samples = np.arange(100)[..., None]
        for i in range(0, 100, 10):
            r.add(samples[i: i + 10])
        samples = r.get_batch(15)
        self.assertEqual(15, samples.shape[0])

    def test_raises_if_get_batch_is_larger_than_total_sample_size(self):
        r = Reservoir(100)
        samples = np.arange(100)[..., None]
        for i in range(0, 100, 10):
            r.add(samples[i: i + 10])
        self.assertRaises(ValueError, lambda: r.get_batch(1000))

    def test_raises_if_get_batch_is_larger_than_available_sample_size(self):
        r = Reservoir(100)
        samples = np.arange(100)[..., None]
        for i in range(0, 50, 10):
            r.add(samples[i: i + 10])
        self.assertRaises(ValueError, lambda: r.get_batch(64))


class TestMultiplexedReservoir(unittest2.TestCase):
    def test_is_consistent_across_keys(self):
        r = MultiplexedReservoir(100)
        samples = np.random.random_sample((10, 3))
        r.add(dict(cat=samples, dog=samples))
        mixed = r.get()
        np.testing.assert_allclose(mixed['cat'], mixed['dog'])

    def test_raises_when_wrong_set_of_keys_passed_to_add(self):
        r = MultiplexedReservoir(100)
        samples = np.random.random_sample((10, 3))
        r.add(dict(cat=samples, dog=samples))
        self.assertRaises(
            ValueError, lambda: r.add(dict(rat=samples, frog=samples)))

    def test_raises_when_some_keys_have_mismatched_lengths(self):
        r = MultiplexedReservoir(100)
        samples = np.random.random_sample((10, 3))
        self.assertRaises(
            ValueError, lambda: r.add(dict(cat=samples, dog=samples[:-1])))

    def test_raises_when_some_keys_have_mismatched_lengths_second_add(self):
        r = MultiplexedReservoir(100)
        samples = np.random.random_sample((10, 3))
        r.add(dict(cat=samples, dog=samples))
        self.assertRaises(
            ValueError, lambda: r.add(dict(cat=samples, dog=samples[:-1])))

    def test_get_returns_dict_with_user_specified_keys(self):
        r = MultiplexedReservoir(100)
        samples = np.random.random_sample((10, 3))
        d = dict(cat=samples, dog=samples)
        r.add(d)
        mixed = r.get()
        self.assertEqual(set(d.keys()), set(mixed.keys()))


class TestReservoirSampler(unittest2.TestCase):
    def test_can_sample_from_one_dimensional_feature(self):
        sampler = ReservoirSampler(nsamples=10)

        frequency_dimension = FrequencyDimension(
            LinearScale(FrequencyBand(100, 1000), 100))

        samples = ArrayWithUnits(
            np.ones((20, 100)),
            [
                TimeDimension(frequency=Seconds(1)),
                frequency_dimension
            ])

        sampler._enqueue(samples, pusher=None)
        reservoir = sampler._r
        self.assertEqual((10, 100), reservoir.shape)
        self.assertIsInstance(reservoir, ArrayWithUnits)
        self.assertEqual(reservoir.dimensions[0], IdentityDimension())
        self.assertEqual(reservoir.dimensions[1], frequency_dimension)

    def test_can_wrap_samples(self):
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
        self.assertIsInstance(reservoir, ArrayWithUnits)
        self.assertEqual(reservoir.dimensions[0], IdentityDimension())
        self.assertEqual(reservoir.dimensions[1], samples.dimensions[1])
        self.assertEqual(reservoir.dimensions[2], samples.dimensions[2])

    def test_can_dequeue_when_reservoir_is_full(self):
        sampler = ReservoirSampler(nsamples=10)

        frequency_dimension = FrequencyDimension(
            LinearScale(FrequencyBand(100, 1000), 100))

        samples = ArrayWithUnits(
            np.ones((10, 10, 100)),
            [
                TimeDimension(frequency=Seconds(10)),
                TimeDimension(frequency=Seconds(1)),
                frequency_dimension
            ])

        sampler._enqueue(samples, pusher=None)
        reservoir = sampler._dequeue()

        self.assertEqual((10, 10, 100), reservoir.shape)
        self.assertIsInstance(reservoir, ArrayWithUnits)
        self.assertEqual(reservoir.dimensions[0], IdentityDimension())
        self.assertEqual(reservoir.dimensions[1], samples.dimensions[1])
        self.assertEqual(reservoir.dimensions[2], samples.dimensions[2])

    def test_can_dequeue_when_reservoir_is_partially_full(self):
        sampler = ReservoirSampler(nsamples=10)

        frequency_dimension = FrequencyDimension(
            LinearScale(FrequencyBand(100, 1000), 100))

        samples = ArrayWithUnits(
            np.ones((4, 10, 100)),
            [
                TimeDimension(frequency=Seconds(10)),
                TimeDimension(frequency=Seconds(1)),
                frequency_dimension
            ])

        sampler._enqueue(samples, pusher=None)
        reservoir = sampler._dequeue()

        self.assertEqual((4, 10, 100), reservoir.shape)
        self.assertIsInstance(reservoir, ArrayWithUnits)
        self.assertEqual(reservoir.dimensions[0], IdentityDimension())
        self.assertEqual(reservoir.dimensions[1], samples.dimensions[1])
        self.assertEqual(reservoir.dimensions[2], samples.dimensions[2])
