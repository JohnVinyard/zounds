import unittest2
import flow
from random_samples import ReservoirSampler
from preprocess import \
    UnitNorm, MeanStdNormalization, PreprocessingPipeline, Pipeline
from learn import LinearRbm
import numpy as np


class Iterator(flow.Node):
    def __init__(self, needs=None):
        super(Iterator, self).__init__(needs=needs)

    def _process(self, data):
        for d in data:
            yield d


class RbmTests(unittest2.TestCase):
    def test_can_retrieve_rbm_pipeline(self):
        class Settings(flow.PersistenceSettings):
            _id = 'rbm'
            id_provider = flow.StaticIdProvider(_id)
            key_builder = flow.StringDelimitedKeyBuilder()
            database = flow.InMemoryDatabase(key_builder=key_builder)

        class Learned(flow.BaseModel, Settings):
            iterator = flow.Feature(
                    Iterator,
                    store=False)

            shuffle = flow.NumpyFeature(
                    ReservoirSampler,
                    nsamples=1000,
                    needs=iterator,
                    store=True)

            unitnorm = flow.PickleFeature(
                    UnitNorm,
                    needs=shuffle,
                    store=False)

            meanstd = flow.PickleFeature(
                    MeanStdNormalization,
                    needs=unitnorm,
                    store=False)

            rbm = flow.PickleFeature(
                    LinearRbm,
                    hdim=64,
                    epochs=5,
                    needs=meanstd,
                    store=False)

            pipeline = flow.PickleFeature(
                    PreprocessingPipeline,
                    needs=(unitnorm, meanstd, rbm),
                    store=True)

        def data():
            for i in xrange(100):
                yield np.random.random_sample((np.random.randint(10, 100), 3))

        Learned.process(iterator=data())
        self.assertIsInstance(Learned().pipeline, Pipeline)
