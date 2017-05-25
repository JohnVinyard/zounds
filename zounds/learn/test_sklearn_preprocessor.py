import unittest2
from sklearn_preprocessor import SklearnModel
from preprocess import PreprocessingPipeline
import featureflow as ff
import numpy as np
from zounds.util import simple_in_memory_settings
from zounds.timeseries import TimeDimension, Milliseconds, Seconds
from zounds.spectral import FrequencyBand, LinearScale, FrequencyDimension
from zounds.core import ArrayWithUnits, IdentityDimension


class MockSklearnModel(object):
    def __init__(self):
        super(MockSklearnModel, self).__init__()
        self.dims = None

    def fit(self, data):
        self.dims = data.shape[1:]
        return self

    def transform(self, data):
        return np.zeros(data.shape)

    def inverse_transform(self, data):
        return np.zeros((data.shape[0],) + self.dims)


@simple_in_memory_settings
class Document(ff.BaseModel):
    l = ff.PickleFeature(
        SklearnModel,
        model=MockSklearnModel())

    pipeline = ff.PickleFeature(
        PreprocessingPipeline,
        needs=(l,),
        store=True)


class SklearnTests(unittest2.TestCase):
    def test_should_handle_forward_transform_of_numpy_array(self):
        training_data = np.zeros((10, 5, 3))
        _id = Document.process(l=training_data)
        doc = Document(_id)
        test_data = np.zeros((11, 5, 3))
        result = doc.pipeline.transform(test_data)
        self.assertEqual((11, 15), result.data.shape)

    def test_should_handle_backward_transform_of_numpy_array(self):
        training_data = np.zeros((10, 5, 3))
        _id = Document.process(l=training_data)
        doc = Document(_id)
        test_data = np.zeros((11, 5, 3))
        result = doc.pipeline.transform(test_data)
        inverted = result.inverse_transform()
        self.assertEqual((11, 5, 3), inverted.shape)

    def test_should_preserve_time_dimension_in_forward_transform(self):
        td = TimeDimension(Seconds(1))
        td2 = TimeDimension(Milliseconds(500))
        fd = FrequencyDimension(LinearScale(FrequencyBand(10, 100), 3))
        training_data = ArrayWithUnits(
            np.zeros((10, 5, 3)), dimensions=(IdentityDimension(), td2, fd))
        _id = Document.process(l=training_data)
        doc = Document(_id)
        test_data = ArrayWithUnits(
            np.zeros((11, 5, 3)), dimensions=(td, td2, fd))
        result = doc.pipeline.transform(test_data)
        self.assertEqual((11, 15), result.data.shape)
        self.assertIsInstance(result.data, ArrayWithUnits)
        self.assertEqual(td, result.data.dimensions[0])
        self.assertEqual(IdentityDimension(), result.data.dimensions[1])

    def test_should_restore_all_dimensions_in_backward_transform(self):
        td = TimeDimension(Seconds(1))
        td2 = TimeDimension(Milliseconds(500))
        fd = FrequencyDimension(LinearScale(FrequencyBand(10, 100), 3))
        training_data = ArrayWithUnits(
            np.zeros((10, 5, 3)), dimensions=(IdentityDimension(), td2, fd))
        _id = Document.process(l=training_data)
        doc = Document(_id)
        test_data = ArrayWithUnits(
            np.zeros((11, 5, 3)), dimensions=(td, td2, fd))
        result = doc.pipeline.transform(test_data)
        inverted = result.inverse_transform()
        self.assertEqual((11, 5, 3), inverted.shape)
        self.assertIsInstance(inverted, ArrayWithUnits)
        self.assertEqual(td, inverted.dimensions[0])
        self.assertEqual(td2, inverted.dimensions[1])
        self.assertEqual(fd, inverted.dimensions[2])