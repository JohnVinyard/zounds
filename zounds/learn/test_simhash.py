import unittest2
from zounds.util import simple_in_memory_settings
import featureflow as ff
from .preprocess import SimHash, PreprocessingPipeline
import numpy as np
from zounds.timeseries import TimeDimension, Seconds
from zounds.spectral import GeometricScale, FrequencyDimension
from zounds.core import IdentityDimension, ArrayWithUnits


@simple_in_memory_settings
class SimHashPipeline(ff.BaseModel):
    simhash = ff.PickleFeature(
        SimHash,
        bits=128)

    pipeline = ff.PickleFeature(
        PreprocessingPipeline,
        needs=(simhash,),
        store=True)


@simple_in_memory_settings
class SimHashPipelineWithPackedBits(ff.BaseModel):
    simhash = ff.PickleFeature(
        SimHash,
        packbits=True,
        bits=1024)

    pipeline = ff.PickleFeature(
        PreprocessingPipeline,
        needs=(simhash,),
        store=True)


class SimHashTests(unittest2.TestCase):
    def test_can_compute_hash_and_pack_bits(self):
        training_data = np.random.normal(0, 1, size=(100, 9))
        _id = SimHashPipelineWithPackedBits.process(simhash=training_data)
        p = SimHashPipelineWithPackedBits(_id)
        test_data = np.random.normal(0, 1, size=(1000, 9))
        result = p.pipeline.transform(test_data).data
        self.assertEqual((1000, 16), result.shape)
        self.assertEqual(np.uint64, result.dtype)

    def test_can_compute_hash_for_1d_vectors(self):
        training_data = np.random.normal(0, 1, size=(100, 9))
        _id = SimHashPipeline.process(simhash=training_data)
        p = SimHashPipeline(_id)
        test_data = np.random.normal(0, 1, size=(1000, 9))
        result = p.pipeline.transform(test_data).data
        self.assertEqual((1000, 128), result.shape)

    def test_can_compute_hash_for_2d_vectors(self):
        training_data = np.random.normal(0, 1, size=(100, 9, 7))
        _id = SimHashPipeline.process(simhash=training_data)
        p = SimHashPipeline(_id)
        test_data = np.random.normal(0, 1, size=(1000, 9, 7))
        result = p.pipeline.transform(test_data).data
        self.assertEqual((1000, 128), result.shape)

    def test_returns_array_with_units_where_possible_1d(self):
        training_data = np.random.normal(0, 1, size=(100, 9))
        _id = SimHashPipeline.process(simhash=training_data)
        p = SimHashPipeline(_id)
        td = TimeDimension(Seconds(1))
        test_data = ArrayWithUnits(
            np.random.normal(0, 1, size=(1000, 9)),
            dimensions=(
                td,
                FrequencyDimension(GeometricScale(20, 20000, 0.1, 9)))
        )
        result = p.pipeline.transform(test_data).data
        self.assertEqual((1000, 128), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(result.dimensions[0], td)
        self.assertIsInstance(result.dimensions[1], IdentityDimension)

    def test_returns_array_with_units_where_possible_2d(self):
        training_data = np.random.normal(0, 1, size=(100, 9, 7))
        _id = SimHashPipeline.process(simhash=training_data)
        p = SimHashPipeline(_id)
        td1 = TimeDimension(Seconds(10))
        td2 = TimeDimension(Seconds(1))
        test_data = ArrayWithUnits(
            np.random.normal(0, 1, size=(1000, 9, 7)),
            dimensions=(
                td1,
                td2,
                FrequencyDimension(GeometricScale(20, 20000, 0.1, 7)))
        )
        result = p.pipeline.transform(test_data).data
        self.assertEqual((1000, 128), result.shape)
        self.assertIsInstance(result, ArrayWithUnits)
        self.assertEqual(result.dimensions[0], td1)
        self.assertIsInstance(result.dimensions[1], IdentityDimension)
