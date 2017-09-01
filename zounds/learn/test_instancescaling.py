import unittest2
from zounds.util import simple_in_memory_settings
import featureflow as ff
from preprocess import InstanceScaling, PreprocessingPipeline
import numpy as np


class InstanceScalingTests(unittest2.TestCase):
    def get_model(self):
        @simple_in_memory_settings
        class Model(ff.BaseModel):
            scaled = ff.PickleFeature(
                InstanceScaling,
                store=False)

            pipeline = ff.PickleFeature(
                PreprocessingPipeline,
                needs=(scaled,),
                store=True)

        training = np.random.random_sample((10, 3))
        _id = Model.process(scaled=training)
        return Model(_id)

    def test_forward_transform_scales_data_2d(self):
        model = self.get_model()
        inp = np.random.random_sample((100, 30)) * 10
        transformed = model.pipeline.transform(inp).data
        self.assertEqual(1.0, transformed.max())

    def test_forward_transform_scales_data_3d(self):
        model = self.get_model()
        inp = np.random.random_sample((100, 30, 3)) - 0.5
        transformed = model.pipeline.transform(inp).data
        self.assertEqual(1.0, np.max(np.abs(transformed)))

    def test_backward_transform_reconstructs_data_2d(self):
        model = self.get_model()
        inp = np.random.random_sample((100, 30)) * 10
        transformed = model.pipeline.transform(inp)
        inverted = transformed.inverse_transform()
        np.testing.assert_allclose(inverted, inp)

    def test_backward_transform_reconstructs_data_3d(self):
        model = self.get_model()
        inp = np.random.random_sample((100, 30, 3)) - 0.5
        transformed = model.pipeline.transform(inp)
        inverted = transformed.inverse_transform()
        np.testing.assert_allclose(inverted, inp)

    def test_correctly_handles_max_of_zero(self):
        model = self.get_model()
        inp = np.random.random_sample((100, 30, 3)) - 0.5
        inp[0, ...] = 0
        transformed = model.pipeline.transform(inp)
        inverted = transformed.inverse_transform()
        np.testing.assert_allclose(inverted, inp)
