import unittest2
import featureflow as ff
import numpy as np

from zounds.util import simple_in_memory_settings
from preprocess import Flatten, PreprocessingPipeline


class FlattenTests(unittest2.TestCase):
    def do_setup(self, shape):
        @simple_in_memory_settings
        class Model(ff.BaseModel):
            flattened = ff.PickleFeature(
                    Flatten,
                    store=False)

            pipeline = ff.PickleFeature(
                    PreprocessingPipeline,
                    needs=(flattened,),
                    store=True)

        training = np.random.random_sample(shape)
        _id = Model.process(flattened=training)
        model = Model(_id)
        data = np.random.random_sample(shape)
        transformed = model.pipeline.transform(data)
        inverted = transformed.inverse_transform()
        return data, transformed.data, inverted

    def test_one_dimensional(self):
        data, transformed, inverted = self.do_setup((10,))
        self.assertEqual((10, 1), transformed.shape)
        self.assertEqual((10,), inverted.shape)
        np.testing.assert_allclose(data.ravel(), transformed.ravel())
        np.testing.assert_allclose(data.ravel(), inverted.ravel())

    def test_two_dimensional(self):
        data, transformed, inverted = self.do_setup((10, 10))
        self.assertEqual((10, 10), transformed.shape)
        self.assertEqual((10, 10), inverted.shape)
        np.testing.assert_allclose(data.ravel(), transformed.ravel())
        np.testing.assert_allclose(data.ravel(), inverted.ravel())

    def test_three_dimensional(self):
        data, transformed, inverted = self.do_setup((10, 10, 10))
        self.assertEqual((10, 100), transformed.shape)
        self.assertEqual((10, 10, 10), inverted.shape)
        np.testing.assert_allclose(data.ravel(), transformed.ravel())
        np.testing.assert_allclose(data.ravel(), inverted.ravel())
