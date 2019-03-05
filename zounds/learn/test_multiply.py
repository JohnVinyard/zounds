import unittest2
import numpy as np
from .preprocess import Multiply, PreprocessingPipeline
from zounds.util import simple_in_memory_settings
import featureflow as ff


class MultiplyTests(unittest2.TestCase):
    def get_model(self, factor):
        @simple_in_memory_settings
        class Model(ff.BaseModel):
            multiply = ff.PickleFeature(
                    Multiply,
                    factor=factor,
                    store=False)

            pipeline = ff.PickleFeature(
                    PreprocessingPipeline,
                    needs=(multiply,),
                    store=True)

        return Model

    def test_can_do_forward_transform_with_scalar(self):
        training = np.random.random_sample((100, 30))
        factor = 10
        Model = self.get_model(factor)
        _id = Model.process(multiply=training)
        model = Model(_id)
        data = np.random.random_sample((10, 30))
        transformed = model.pipeline.transform(data)
        np.testing.assert_allclose(data * factor, transformed.data)

    def test_can_do_forward_transform_with_array(self):
        training = np.random.random_sample((100, 30))
        factor = np.random.random_sample(30)
        Model = self.get_model(factor)
        _id = Model.process(multiply=training)
        model = Model(_id)
        data = np.random.random_sample((10, 30))
        transformed = model.pipeline.transform(data)
        np.testing.assert_allclose(data * factor, transformed.data)

    def test_raises_if_shapes_do_not_match(self):
        training = np.random.random_sample((100, 30))
        factor = np.random.random_sample(3)
        Model = self.get_model(factor)
        self.assertRaises(ValueError, lambda: Model.process(multiply=training))

    def test_can_do_forward_and_backward_transform_with_scalar(self):
        training = np.random.random_sample((100, 30))
        factor = 10
        Model = self.get_model(factor)
        _id = Model.process(multiply=training)
        model = Model(_id)
        data = np.random.random_sample((10, 30))
        transformed = model.pipeline.transform(data)
        recon = transformed.inverse_transform()
        np.testing.assert_allclose(data, recon)

    def test_can_do_forward_and_backward_transform_with_array(self):
        training = np.random.random_sample((100, 30))
        factor = np.random.random_sample(30)
        Model = self.get_model(factor)
        _id = Model.process(multiply=training)
        model = Model(_id)
        data = np.random.random_sample((10, 30))
        transformed = model.pipeline.transform(data)
        recon = transformed.inverse_transform()
        np.testing.assert_allclose(data, recon)
