import unittest2
import featureflow as ff
import numpy as np

from zounds.util import simple_in_memory_settings
from .preprocess import Reshape, PreprocessingPipeline


class ReshapeTests(unittest2.TestCase):

    def do_setup(self, shape, new_shape):
        @simple_in_memory_settings
        class Model(ff.BaseModel):
            flattened = ff.PickleFeature(
                    Reshape,
                    new_shape=new_shape,
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

    def do_assertions(self, original_shape, new_shape, expected_shape):
        data, transformed, inverted = self.do_setup(original_shape, new_shape)
        self.assertEqual(expected_shape, transformed.shape)
        self.assertEqual(original_shape, inverted.shape)
        np.testing.assert_allclose(data.ravel(), transformed.ravel())
        np.testing.assert_allclose(data.ravel(), inverted.ravel())

    def test_reshape(self):
        self.do_assertions(
            original_shape=(10, 100),
            new_shape=(10, 10),
            expected_shape=(10, 10, 10))

    def test_expand_multiple_dimensions(self):
        self.do_assertions(
            original_shape=(10, 40),
            new_shape=(1, -1, 1),
            expected_shape=(10, 1, 40, 1))

    def test_expand_middle_dimension(self):
        self.do_assertions(
            original_shape=(10, 40),
            new_shape=(1, -1),
            expected_shape=(10, 1, 40))

    def test_expand_last_dimension(self):
        self.do_assertions(
            original_shape=(10, 40),
            new_shape=(-1, 1),
            expected_shape=(10, 40, 1))

    def test_flatten_one_dimensional(self):
        self.do_assertions(
            original_shape=(10,),
            new_shape=tuple(),
            expected_shape=(10,))

    def test_flatten_two_dimensional(self):
        self.do_assertions(
            original_shape=(10, 40),
            new_shape=(-1,),
            expected_shape=(10, 40))

    def test_flatten_three_dimensional(self):
        self.do_assertions(
            original_shape=(10, 40, 20),
            new_shape=(-1,),
            expected_shape=(10, 800))
