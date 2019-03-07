import unittest2
import featureflow
from .random_samples import ReservoirSampler
from .preprocess import \
    UnitNorm, MeanStdNormalization, PreprocessingPipeline, Pipeline
from .learn import Learned, KMeans
import numpy as np


class Iterator(featureflow.Node):
    def __init__(self, needs=None):
        super(Iterator, self).__init__(needs=needs)

    def _process(self, data):
        for d in data:
            yield d


def build_classes():
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
            KMeans,
            centroids=3,
            needs=meanstd,
            store=False)

        pipeline = featureflow.PickleFeature(
            PreprocessingPipeline,
            needs=(unitnorm, meanstd, rbm),
            store=True)

    return Rbm


def data():
    for i in range(100):
        yield np.random.random_sample((np.random.randint(10, 100), 9))


class RbmTests(unittest2.TestCase):
    def test_can_retrieve_rbm_pipeline(self):
        KMeans = build_classes()
        KMeans.process(iterator=data())
        self.assertIsInstance(KMeans().pipeline, Pipeline)


class LearnedTests(unittest2.TestCase):
    def test_can_use_learned_feature(self):
        KMeans = build_classes()
        KMeans.process(iterator=data())
        l = Learned(learned=KMeans())
        results = list(l._process(np.random.random_sample((33, 9))))[0]
        self.assertEqual((33, 3), results.shape)

    def test_pipeline_changes_version_when_recomputed(self):
        KMeans = build_classes()
        KMeans.process(iterator=data())
        v1 = Learned(learned=KMeans()).version
        v2 = Learned(learned=KMeans()).version
        self.assertEqual(v1, v2)
        KMeans.process(iterator=data())
        v3 = Learned(learned=KMeans()).version
        self.assertNotEqual(v1, v3)

    def test_pipeline_does_not_store_computed_data_from_training(self):
        Rbm = build_classes()
        Rbm.process(iterator=data())
        rbm = Rbm()
        pipeline_data = rbm.pipeline.processors[-1].data
        self.assertIsNone(pipeline_data)

    def test_can_pass_a_pipeline_slice_to_be_applied_at_inference_time(self):
        KMeans = build_classes()
        KMeans.process(iterator=data())
        l = Learned(learned=KMeans(), pipeline_func=lambda x: x.pipeline[:2])
        results = list(l._process(np.random.random_sample((33, 9))))[0]
        self.assertEqual((33, 9), results.shape)
