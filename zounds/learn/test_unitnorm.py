import featureflow as ff
import unittest2
import numpy as np

from zounds.util import simple_in_memory_settings
from preprocess import UnitNorm, PreprocessingPipeline
from learn import KMeans
from zounds.timeseries import ConstantRateTimeSeries, Seconds
from zounds.spectral import \
    TimeFrequencyRepresentation, FrequencyBand, LinearScale


class UnitNormTests(unittest2.TestCase):

    def get_model(self):
        @simple_in_memory_settings
        class Model(ff.BaseModel):

            unitnorm = ff.PickleFeature(
                UnitNorm,
                store=False)

            kmeans = ff.PickleFeature(
                KMeans,
                centroids=10,
                needs=unitnorm,
                store=False)

            pipeline = ff.PickleFeature(
                PreprocessingPipeline,
                needs=(unitnorm, kmeans),
                store=True)

        return Model

    def invert_and_assert_class(self, data):
        training = np.random.random_sample((100, 30))
        Model = self.get_model()
        _id = Model.process(unitnorm=training)
        model = Model(_id)
        transformed = model.pipeline.transform(data)
        inverted = transformed.inverse_transform()
        self.assertEqual(data.__class__, inverted.__class__)
        return inverted

    def test_inversion_returns_array(self):
        data = np.random.random_sample((10, 30))
        self.invert_and_assert_class(data)

    def test_inversion_returns_time_series(self):
        data = np.random.random_sample((33, 30))
        ts = ConstantRateTimeSeries(data, frequency=Seconds(1))
        inverted = self.invert_and_assert_class(ts)
        self.assertEqual(Seconds(1), inverted.frequency)

    def test_inversion_returns_time_frequency_representation(self):
        data = np.random.random_sample((33, 30))
        scale = LinearScale(FrequencyBand(20, 20000), 30)
        tf = TimeFrequencyRepresentation(
            data,
            frequency=Seconds(1),
            duration=Seconds(2),
            scale=scale)
        inverted = self.invert_and_assert_class(tf)
        self.assertEqual(Seconds(1), inverted.frequency)
        self.assertEqual(Seconds(2), inverted.duration)
        self.assertEqual(scale, inverted.scale)
