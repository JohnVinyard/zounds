import unittest2
import featureflow as ff
from preprocess import Weighted, InstanceScaling, PreprocessingPipeline
from zounds.spectral import \
    AWeighting, GeometricScale, FrequencyDimension, FrequencyAdaptive
from zounds.core import ArrayWithUnits
from zounds.timeseries import TimeDimension, Seconds
from zounds.util import simple_in_memory_settings
import numpy as np


class WeightedTests(unittest2.TestCase):

    def test_can_apply_weighting(self):

        @simple_in_memory_settings
        class Model(ff.BaseModel):
            weighted = ff.PickleFeature(
                Weighted,
                weighting=AWeighting(),
                store=False)

            scaled = ff.PickleFeature(
                InstanceScaling,
                needs=weighted,
                store=False)

            pipeline = ff.PickleFeature(
                PreprocessingPipeline,
                needs=(weighted, scaled),
                store=True)

        training = ArrayWithUnits(
            np.random.random_sample((100, 15)),
            dimensions=[
                TimeDimension(Seconds(1)),
                FrequencyDimension(GeometricScale(100, 1000, 0.1, 15))
            ]
        )

        _id = Model.process(weighted=training)
        model = Model(_id)

        test = ArrayWithUnits(
            np.random.random_sample((10, 15)),
            dimensions=[
                TimeDimension(Seconds(1)),
                FrequencyDimension(GeometricScale(100, 1000, 0.1, 15))
            ]
        )

        result = model.pipeline.transform(test)
        inverted = result.inverse_transform()

        np.testing.assert_allclose(test, inverted)

    def test_can_apply_weighting_to_frequency_adaptive_transform(self):
        @simple_in_memory_settings
        class Model(ff.BaseModel):
            weighted = ff.PickleFeature(
                Weighted,
                weighting=AWeighting(),
                store=False)

            scaled = ff.PickleFeature(
                InstanceScaling,
                needs=weighted,
                store=False)

            pipeline = ff.PickleFeature(
                PreprocessingPipeline,
                needs=(weighted, scaled),
                store=True)

        scale = GeometricScale(100, 1000, 0.1, 15)

        training = FrequencyAdaptive(
            [np.ones((100, x)) for x in xrange(1, len(scale) + 1)],
            time_dimension=TimeDimension(Seconds(1)),
            scale=scale)

        _id = Model.process(weighted=training)
        model = Model(_id)

        test = FrequencyAdaptive(
            [np.ones((10, x)) for x in xrange(1, len(scale) + 1)],
            time_dimension=TimeDimension(Seconds(1)),
            scale=scale)

        result = model.pipeline.transform(test)
        inverted = result.inverse_transform()

        self.assertIsInstance(inverted, FrequencyAdaptive)
        np.testing.assert_allclose(test, inverted)


