import unittest2
import featureflow
from random_samples import ReservoirSampler
from preprocess import \
    UnitNorm, MeanStdNormalization, PreprocessingPipeline, Pipeline
from learn import LinearRbm, Learned
import numpy as np


class Iterator(featureflow.Node):
    def __init__(self, needs=None):
        super(Iterator, self).__init__(needs=needs)

    def _process(self, data):
        for d in data:
            yield d


class Settings(featureflow.PersistenceSettings):
    _id = 'rbm'
    id_provider = featureflow.StaticIdProvider(_id)
    key_builder = featureflow.StringDelimitedKeyBuilder()
    database = featureflow.InMemoryDatabase(key_builder=key_builder)


class Rbm(featureflow.BaseModel, Settings):
    iterator = featureflow.Feature(
            Iterator,
            store=False)

    shuffle = featureflow.NumpyFeature(
            ReservoirSampler,
            nsamples=1000,
            needs=iterator,
            store=True)

    unitnorm = featureflow.PickleFeature(
            UnitNorm,
            needs=shuffle,
            store=False)

    meanstd = featureflow.PickleFeature(
            MeanStdNormalization,
            needs=unitnorm,
            store=False)

    rbm = featureflow.PickleFeature(
            LinearRbm,
            hdim=64,
            epochs=5,
            needs=meanstd,
            store=False)

    pipeline = featureflow.PickleFeature(
            PreprocessingPipeline,
            needs=(unitnorm, meanstd, rbm),
            store=True)


def data():
    for i in xrange(100):
        yield np.random.random_sample((np.random.randint(10, 100), 3))


class RbmTests(unittest2.TestCase):
    def test_can_retrieve_rbm_pipeline(self):
        Rbm.process(iterator=data())
        self.assertIsInstance(Rbm().pipeline, Pipeline)


class LearnedTests(unittest2.TestCase):
    def test_can_use_learned_feature(self):
        Rbm.process(iterator=data())
        l = Learned(learned=Rbm())
        results = list(l._process(np.random.random_sample((33, 3))))[0]
        self.assertEqual((33, 64), results.shape)

    def test_pipeline_changes_version_when_recomputed(self):
        Rbm.process(iterator=data())
        v1 = Learned(learned=Rbm()).version
        v2 = Learned(learned=Rbm()).version
        self.assertEqual(v1, v2)
        Rbm.process(iterator=data())
        v3 = Learned(learned=Rbm()).version
        self.assertNotEqual(v1, v3)


