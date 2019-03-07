import unittest2
from zounds.util import simple_in_memory_settings
from .preprocess import MeanStdNormalization, PreprocessingPipeline
import featureflow as ff
import numpy as np


class MeanStdTests(unittest2.TestCase):
    def _forward_backward(self, shape):
        @simple_in_memory_settings
        class Model(ff.BaseModel):
            meanstd = ff.PickleFeature(
                    MeanStdNormalization,
                    store=False)

            pipeline = ff.PickleFeature(
                    PreprocessingPipeline,
                    needs=(meanstd,),
                    store=True)

        training = np.random.random_sample((100,) + shape)
        _id = Model.process(meanstd=training)
        model = Model(_id)

        data_shape = (10,) + shape
        data = np.random.random_sample(data_shape)
        result = model.pipeline.transform(data)
        self.assertEqual(data_shape, result.data.shape)
        inverted = result.inverse_transform()
        self.assertEqual(inverted.shape, data.shape)
        np.testing.assert_allclose(inverted, data)

    def test_can_process_1d(self):
        self._forward_backward((9,))

    def test_can_process_2d(self):
        self._forward_backward((3, 4))

    def test_can_process_3d(self):
        self._forward_backward((5, 4, 7))
