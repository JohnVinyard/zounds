import unittest2
import featureflow
import numpy as np
from preprocess import \
    UnitNorm, MeanStdNormalization, Binarize, PreprocessingPipeline


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
