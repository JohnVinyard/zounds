import featureflow
import numpy as np
import unittest2

from preprocess import \
    UnitNorm, MeanStdNormalization, Binarize, PreprocessingPipeline, Log
from zounds.spectral import \
    GeometricScale, FrequencyAdaptive
from zounds.timeseries import Seconds, TimeDimension
from zounds.util import simple_in_memory_settings


class TestPipeline(unittest2.TestCase):
    def test_cannot_invert_pipeline_if_any_steps_are_missing(self):
        class Settings(featureflow.PersistenceSettings):
            id_provider = featureflow.UuidProvider()
            key_builder = featureflow.StringDelimitedKeyBuilder()
            database = featureflow.InMemoryDatabase(key_builder=key_builder)

        class Model(featureflow.BaseModel, Settings):
            unitnorm = featureflow.PickleFeature(
                UnitNorm,
                store=False)

            binary = featureflow.PickleFeature(
                Binarize,
                needs=unitnorm,
                store=False)

            pipeline = featureflow.PickleFeature(
                PreprocessingPipeline,
                needs=(unitnorm, binary),
                store=True)

        data = np.random.random_sample((1000, 4))
        _id = Model.process(unitnorm=data)
        example = np.random.random_sample((10, 4))
        model = Model(_id)
        transformed = model.pipeline.transform(example)
        self.assertRaises(
            NotImplementedError, lambda: transformed.inverse_transform())

    def test_can_invert_pipeline(self):
        class Settings(featureflow.PersistenceSettings):
            id_provider = featureflow.UuidProvider()
            key_builder = featureflow.StringDelimitedKeyBuilder()
            database = featureflow.InMemoryDatabase(key_builder=key_builder)

        class Model(featureflow.BaseModel, Settings):
            unitnorm = featureflow.PickleFeature(
                UnitNorm,
                store=False)

            meanstd = featureflow.PickleFeature(
                MeanStdNormalization,
                needs=unitnorm,
                store=False)

            pipeline = featureflow.PickleFeature(
                PreprocessingPipeline,
                needs=(unitnorm, meanstd),
                store=True)

        data = np.random.random_sample((1000, 4))
        _id = Model.process(unitnorm=data)
        example = np.random.random_sample((10, 4))
        model = Model(_id)
        transformed = model.pipeline.transform(example)
        reconstructed = transformed.inverse_transform()
        diff = np.abs(example - reconstructed)
        self.assertTrue(np.all(diff < .00001))

    def test_can_invert_pipeline_with_log(self):
        class Settings(featureflow.PersistenceSettings):
            id_provider = featureflow.UuidProvider()
            key_builder = featureflow.StringDelimitedKeyBuilder()
            database = featureflow.InMemoryDatabase(key_builder=key_builder)

        class Model(featureflow.BaseModel, Settings):
            log = featureflow.PickleFeature(
                Log,
                store=False)

            meanstd = featureflow.PickleFeature(
                MeanStdNormalization,
                needs=log,
                store=False)

            pipeline = featureflow.PickleFeature(
                PreprocessingPipeline,
                needs=(log, meanstd),
                store=True)

        data = np.random.random_sample((1000, 4))
        _id = Model.process(log=data)
        example = np.random.random_sample((10, 4))
        model = Model(_id)
        transformed = model.pipeline.transform(example)
        reconstructed = transformed.inverse_transform()
        diff = np.abs(example - reconstructed)
        self.assertTrue(np.all(diff < .00001))

    def test_can_invert_pipeline_that_takes_frequency_adaptive_transform(self):
        td = TimeDimension(frequency=Seconds(1))
        scale = GeometricScale(20, 5000, 0.05, 10)
        arrs = [np.zeros((10, x)) for x in xrange(1, 11)]
        fa = FrequencyAdaptive(arrs, td, scale)

        @simple_in_memory_settings
        class Model(featureflow.BaseModel):
            log = featureflow.PickleFeature(
                Log,
                store=False)

            meanstd = featureflow.PickleFeature(
                MeanStdNormalization,
                needs=log,
                store=False)

            pipeline = featureflow.PickleFeature(
                PreprocessingPipeline,
                needs=(log, meanstd),
                store=True)

        _id = Model.process(log=fa)
        model = Model(_id)
        result = model.pipeline.transform(fa)
        recon = result.inverse_transform()
        self.assertIsInstance(recon, FrequencyAdaptive)
        self.assertEqual(fa.shape, recon.shape)