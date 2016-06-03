from random import random

import featureflow as ff
import numpy as np
import unittest2

from duration import Milliseconds
from timeseries import TimeSlice
from variablerate import VariableRateTimeSeries, VariableRateTimeSeriesFeature
from zounds.basic import Pooled, stft
from zounds.segment import TimeSliceFeature
from zounds.synthesize import NoiseSynthesizer
from zounds.timeseries import Picoseconds, Seconds, SR44100


class VariableRateTimeSeriesFeatureTests(unittest2.TestCase):
    def test_can_encode_and_decode_variable_rate_time_Series(self):

        class TimestampEmitter(ff.Node):
            def __init__(self, needs=None):
                super(TimestampEmitter, self).__init__(needs=needs)
                self.pos = Picoseconds(0)

            def _process(self, data):
                for i, d in enumerate(data):
                    if random() > 0.9:
                        yield self.pos + (i * data.frequency)
                self.pos += data.frequency * len(data)

        class Settings(ff.PersistenceSettings):
            id_provider = ff.UuidProvider()
            key_builder = ff.StringDelimitedKeyBuilder()
            database = ff.InMemoryDatabase(key_builder=key_builder)

        graph = stft(store_fft=True)

        class Document(graph, Settings):
            slices = TimeSliceFeature(
                    TimestampEmitter,
                    needs=graph.fft,
                    store=True)

            pooled = VariableRateTimeSeriesFeature(
                    Pooled,
                    op=np.max,
                    axis=0,
                    needs=(slices, graph.fft),
                    store=False)

        signal = NoiseSynthesizer(SR44100()).synthesize(Seconds(10)).encode()
        _id = Document.process(meta=signal)
        doc = Document(_id)
        self.assertIsInstance(doc.pooled, VariableRateTimeSeries)
        self.assertEqual(doc.fft.shape[1], doc.pooled.slicedata.shape[1])


class VariableRateTimeSeriesTests(unittest2.TestCase):
    def test_can_slice_time_series_with_time_slice(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10)),
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10))
        ))
        sliced = ts[TimeSlice(start=Seconds(1), duration=Seconds(2))]
        self.assertIsInstance(sliced, VariableRateTimeSeries)
        self.assertEqual(1, len(sliced))

    def test_can_slice_time_series_with_open_ended_time_slice(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10)),
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10))
        ))
        sliced = ts[TimeSlice(start=Seconds(1))]
        self.assertIsInstance(sliced, VariableRateTimeSeries)
        self.assertEqual(2, len(sliced))

    def test_can_index_time_series_with_integer_index(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10)),
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10))
        ))
        sliced = ts[1]
        self.assertIsInstance(sliced, np.record)
        timeslice, data = sliced
        self.assertEqual(
                TimeSlice(start=Seconds(1), duration=Seconds(2)), timeslice)
        self.assertIsInstance(data, np.ndarray)
        self.assertEqual((10,), data.shape)

    def test_can_slice_time_series_with_integer_indices(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10)),
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10))
        ))
        sliced = ts[1:]
        self.assertIsInstance(sliced, VariableRateTimeSeries)
        self.assertEqual(2, len(sliced))

    def test_get_index_error_when_using_out_of_range_int_index(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10)),
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10))
        ))
        self.assertRaises(IndexError, lambda: ts[5])

    def test_get_empty_time_series_when_using_out_of_range_time_slice(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10)),
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10))
        ))
        sliced = ts[TimeSlice(start=Seconds(10), duration=Seconds(1))]
        self.assertIsInstance(sliced, VariableRateTimeSeries)
        self.assertEqual(0, len(sliced))

    def test_time_slice_spanning_less_than_one_example_returns_one_example(
            self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10)),
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10))
        ))
        timeslice = TimeSlice(
                start=Milliseconds(500), duration=Milliseconds(100))
        sliced = ts[timeslice]
        self.assertIsInstance(sliced, VariableRateTimeSeries)
        self.assertEqual(1, len(sliced))

    def test_time_slice_spanning_multiple_examples_returns_all_examples(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10)),
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10))
        ))
        timeslice = TimeSlice(start=Milliseconds(500), duration=Seconds(1))
        sliced = ts[timeslice]
        self.assertIsInstance(sliced, VariableRateTimeSeries)
        self.assertEqual(2, len(sliced))

    def test_can_get_entire_time_series_with_empty_slice(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10)),
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10))
        ))
        sliced = ts[:]
        self.assertIsInstance(sliced, VariableRateTimeSeries)
        self.assertEqual(3, len(sliced))

    def test_sorts_input(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10))
        ))
        timeslice, data = ts[0]
        self.assertEqual(
                TimeSlice(start=Seconds(0), duration=Seconds(1)), timeslice)

    def raises_if_data_is_of_variable_size(self):
        data = ((
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(11)),
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10))
        ))
        self.assertRaises(ValueError, VariableRateTimeSeries(data))

    def test_span(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10)),
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10))
        ))
        self.assertEqual(
                TimeSlice(start=Seconds(0), duration=Seconds(4)), ts.span)

    def test_span_empty(self):
        ts = VariableRateTimeSeries(())
        self.assertEqual(
                TimeSlice(start=Seconds(0), duration=Seconds(0)), ts.span)

    def test_end(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(0), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(1), duration=Seconds(2)), np.zeros(10)),
            (TimeSlice(start=Seconds(3), duration=Seconds(1)), np.zeros(10))
        ))
        self.assertEqual(Seconds(4), ts.end)

    def test_end_empty(self):
        ts = VariableRateTimeSeries(())
        self.assertEqual(Seconds(0), ts.end)

