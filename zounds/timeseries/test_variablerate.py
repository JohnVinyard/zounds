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
from zounds.util import simple_in_memory_settings


class VariableRateTimeSeriesFeatureTests(unittest2.TestCase):

    def test_can_encode_and_decode_variable_rate_time_Series(self):

        class TimestampEmitter(ff.Node):
            def __init__(self, needs=None):
                super(TimestampEmitter, self).__init__(needs=needs)
                self.pos = Picoseconds(0)

            def _process(self, data):
                td = data.dimensions[0]
                frequency = td.frequency
                timestamps = [self.pos + (i * frequency)
                        for i, d in enumerate(data)
                        if random() > 0.9]
                slices = TimeSlice.slices(timestamps)
                yield VariableRateTimeSeries(
                    (ts, np.zeros(0)) for ts in slices)
                self.pos += frequency * len(data)

        graph = stft(store_fft=True)

        @simple_in_memory_settings
        class Document(graph):
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

        signal = NoiseSynthesizer(SR44100())\
            .synthesize(Seconds(10))\
            .encode()
        _id = Document.process(meta=signal)
        doc = Document(_id)
        self.assertIsInstance(doc.pooled, VariableRateTimeSeries)
        self.assertEqual(doc.fft.shape[1], doc.pooled.slicedata.shape[1])


class VariableRateTimeSeriesTests(unittest2.TestCase):

    def test_can_concatenate_variable_rate_time_series(self):
        ts1 = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(1), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(2), duration=Seconds(1)), np.zeros(10)),
        ))
        ts2 = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(1), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(2), duration=Seconds(1)), np.zeros(10)),
        ))
        ts3 = ts1.concat(ts2)
        self.assertEqual(4, len(ts3))
        self.assertEqual((4, 10), ts3.slicedata.shape)

    def test_concat_fails_when_data_shape_is_mismatched(self):
        ts1 = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(1), duration=Seconds(1)), np.zeros(10)),
            (TimeSlice(start=Seconds(2), duration=Seconds(1)), np.zeros(10)),
        ))
        ts2 = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(1), duration=Seconds(1)), np.zeros(11)),
            (TimeSlice(start=Seconds(2), duration=Seconds(1)), np.zeros(11)),
        ))
        self.assertRaises(ValueError, lambda: ts1.concat(ts2))

    def test_can_create_instance_with_no_slice_data(self):
        ts = VariableRateTimeSeries((
            (TimeSlice(start=Seconds(1), duration=Seconds(1)), np.zeros(0)),
            (TimeSlice(start=Seconds(2), duration=Seconds(1)), np.zeros(0)),
        ))
        self.assertEqual(2, len(ts))
        self.assertEqual((2, 0), ts.slicedata.shape)

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

