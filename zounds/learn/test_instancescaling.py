import unittest2
from zounds.util import simple_in_memory_settings
import featureflow as ff
from preprocess import InstanceScaling, PreprocessingPipeline
import numpy as np


class InstanceScalingTests(unittest2.TestCase):

    def get_model(self, max_value=1):
        @simple_in_memory_settings
        class Model(ff.BaseModel):
            scaled = ff.PickleFeature(
                InstanceScaling,
                max_value=max_value,
                store=False)

            pipeline = ff.PickleFeature(
                PreprocessingPipeline,
                needs=(scaled,),
                store=True)

        training = np.random.random_sample((10, 3))
        _id = Model.process(scaled=training)
        return Model(_id)

    def test_can_scale_to_arbitrary_value(self):
        model = self.get_model(max_value=50)
        inp = np.random.random_sample((100, 30)) * 10
        transformed = model.pipeline.transform(inp).data
        self.assertEqual(50.0, transformed.max())

    def test_can_invert_when_scaling_to_arbitrary_value(self):
        model = self.get_model(max_value=50)
        inp = np.random.random_sample((100, 30)) * 10
        transformed = model.pipeline.transform(inp)
        inverted = transformed.inverse_transform()
        np.testing.assert_allclose(inverted, inp)

    def test_forward_transform_scales_data_2d(self):
        model = self.get_model()
        inp = np.random.random_sample((100, 30)) * 10
        transformed = model.pipeline.transform(inp).data
        self.assertEqual(1.0, transformed.max())

    def test_each_example_has_same_max(self):
        model = self.get_model()
        inp = np.random.normal(0, 1, (100, 30)) * 50
        transformed = model.pipeline.transform(inp).data
        np.testing.assert_allclose(np.abs(transformed).max(axis=-1), 1)

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

    def test_instance_scaling_maintains_dtype(self):
        model = self.get_model()
        inp = np.random.random_sample((100, 30)) * 10
        inp = inp.astype(np.float32)
        transformed = model.pipeline.transform(inp).data
        self.assertEqual(np.float32, transformed.dtype)
