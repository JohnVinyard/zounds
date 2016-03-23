import unittest2
import numpy as np
from featureflow import BaseModel, Node, PersistenceSettings
from basic import Merge
from zounds.timeseries import \
    ConstantRateTimeSeries, ConstantRateTimeSeriesFeature, Milliseconds


class MergeTester(Node):
    def __init__(
            self,
            total_frames=100,
            increments_of=30,
            features=10,
            needs=None):
        super(MergeTester, self).__init__(needs=needs)
        self.features = features
        self.increments_of = increments_of
        self.total_frames = total_frames

    def _process(self, data):
        for i in xrange(0, self.total_frames, self.increments_of):
            size = min(self.increments_of, self.total_frames - i)
            yield ConstantRateTimeSeries(
                    np.zeros((size, self.features)),
                    frequency=Milliseconds(500))


class MergeTests(unittest2.TestCase):
    def test_raises_if_single_source(self):
        class Document(BaseModel, PersistenceSettings):
            source = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    store=True)
            merged = ConstantRateTimeSeriesFeature(
                    Merge,
                    needs=source,
                    store=True)

        self.assertRaises(ValueError, lambda: Document.process(source=''))

    def test_raises_if_single_element_iterable(self):
        class Document(BaseModel, PersistenceSettings):
            source = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    store=True)
            merged = ConstantRateTimeSeriesFeature(
                    Merge,
                    needs=[source],
                    store=True)

        self.assertRaises(ValueError, lambda: Document.process(source=''))

    def test_can_combine_two_sources_at_same_rate(self):
        class Document(BaseModel, PersistenceSettings):
            source1 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    store=True)
            source2 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    store=True)
            merged = ConstantRateTimeSeriesFeature(
                    Merge,
                    needs=[source1, source2],
                    store=True)

        _id = Document.process(source1='', source2='')
        doc = Document(_id)
        self.assertEqual((200, 20), doc.merged.shape)

    def test_can_combine_three_sources_at_same_rate(self):
        class Document(BaseModel, PersistenceSettings):
            source1 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    store=True)
            source2 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    store=True)
            source3 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    store=True)
            merged = ConstantRateTimeSeriesFeature(
                    Merge,
                    needs=(source1, source2, source3),
                    store=True)

        _id = Document.process(source1='', source2='', source3='')
        doc = Document(_id)
        self.assertEqual((200, 30), doc.merged.shape)

    def test_can_combine_two_sources_at_different_rates(self):
        class Document(BaseModel, PersistenceSettings):
            source1 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    increments_of=30,
                    store=True)
            source2 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    increments_of=40,
                    store=True)
            merged = ConstantRateTimeSeriesFeature(
                    Merge,
                    needs=(source1, source2),
                    store=True)

        _id = Document.process(source1='', source2='')
        doc = Document(_id)
        self.assertEqual((200, 20), doc.merged.shape)

    def test_can_combine_three_sources_at_different_rates(self):
        class Document(BaseModel, PersistenceSettings):
            source1 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    increments_of=12,
                    store=True)
            source2 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    increments_of=17,
                    store=True)
            source3 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    increments_of=32,
                    store=True)
            merged = ConstantRateTimeSeriesFeature(
                    Merge,
                    needs=(source1, source2, source3),
                    store=True)

        _id = Document.process(source1='', source2='', source3='')
        doc = Document(_id)
        self.assertEqual((200, 30), doc.merged.shape)

    def test_shortest_of_two_sources(self):
        class Document(BaseModel, PersistenceSettings):
            source1 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=190,
                    increments_of=30,
                    store=True)
            source2 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    increments_of=40,
                    store=True)
            merged = ConstantRateTimeSeriesFeature(
                    Merge,
                    needs=[source1, source2],
                    store=True)

        _id = Document.process(source1='', source2='')
        doc = Document(_id)
        self.assertEqual((190, 20), doc.merged.shape)

    def test_shortest_of_three_sources(self):
        class Document(BaseModel, PersistenceSettings):
            source1 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=200,
                    increments_of=12,
                    store=True)
            source2 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=185,
                    increments_of=17,
                    store=True)
            source3 = ConstantRateTimeSeriesFeature(
                    MergeTester,
                    total_frames=50,
                    increments_of=32,
                    store=True)
            merged = ConstantRateTimeSeriesFeature(
                    Merge,
                    needs=[source1, source2, source3],
                    store=True)

        _id = Document.process(source1='', source2='', source3='')
        doc = Document(_id)
        self.assertEqual((50, 30), doc.merged.shape)
