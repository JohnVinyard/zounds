import unittest2
import flow
import numpy as np
from preprocess import \
    UnitNorm, MeanStdNormalization, Binarize, PreprocessingPipeline


class TestPipeline(unittest2.TestCase):
    def test_cannot_invert_pipeline_if_any_steps_are_missing(self):
        class Settings(flow.PersistenceSettings):
            id_provider = flow.UuidProvider()
            key_builder = flow.StringDelimitedKeyBuilder()
            database = flow.InMemoryDatabase(key_builder=key_builder)

        class Model(flow.BaseModel, Settings):
            unitnorm = flow.PickleFeature(
                    UnitNorm,
                    store=False)

            binary = flow.PickleFeature(
                    Binarize,
                    needs=unitnorm,
                    store=False)

            pipeline = flow.PickleFeature(
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
        class Settings(flow.PersistenceSettings):
            id_provider = flow.UuidProvider()
            key_builder = flow.StringDelimitedKeyBuilder()
            database = flow.InMemoryDatabase(key_builder=key_builder)

        class Model(flow.BaseModel, Settings):
            unitnorm = flow.PickleFeature(
                    UnitNorm,
                    store=False)

            meanstd = flow.PickleFeature(
                    MeanStdNormalization,
                    needs=unitnorm,
                    store=False)

            pipeline = flow.PickleFeature(
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
