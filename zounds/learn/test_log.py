import unittest2
from zounds.util import simple_in_memory_settings
import featureflow as ff
from .preprocess import Log, PreprocessingPipeline
import numpy as np


class LogTests(unittest2.TestCase):
    def get_model(self):
        @simple_in_memory_settings
        class Model(ff.BaseModel):
            log = ff.PickleFeature(
                    Log,
                    store=False)

            pipeline = ff.PickleFeature(
                    PreprocessingPipeline,
                    needs=(log,),
                    store=True)

        training = np.random.random_sample((10, 3))
        _id = Model.process(log=training)
        return Model(_id)

    def test_forward_transform_preserves_sign_large_values(self):
        model = self.get_model()
        inp = (np.random.random_sample((100, 3)) * 1000) - 500
        s1 = np.sign(inp)
        result = model.pipeline.transform(inp)
        s2 = np.sign(result.data)
        self.assertTrue(np.all(s1 == s2))

    def test_forward_transform_preserves_sign(self):
        model = self.get_model()
        inp = np.random.random_sample((100, 3)) - 0.5
        s1 = np.sign(inp)
        result = model.pipeline.transform(inp)
        s2 = np.sign(result.data)
        self.assertTrue(np.all(s1 == s2))

    def test_backward_transform_reconstructs_data_large_values(self):
        model = self.get_model()
        inp = (np.random.random_sample((100, 3)) * 1000) - 500
        result = model.pipeline.transform(inp)
        inverted = result.inverse_transform()
        np.testing.assert_allclose(inp, inverted)

    def test_backward_transform_reconstructs_data(self):
        model = self.get_model()
        inp = np.random.random_sample((100, 3)) - 0.5
        result = model.pipeline.transform(inp)
        inverted = result.inverse_transform()
        np.testing.assert_allclose(inp, inverted)

    def test_pipeline_requires_no_inversion_data(self):
        model = self.get_model()
        inp = (np.random.random_sample((100, 3)) * 1000) - 500
        result = model.pipeline.transform(inp)
        self.assertFalse(result.inversion_data[0])
