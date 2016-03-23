import unittest
import numpy as np
from npx import windowed, sliding_window, downsample, Growable


class GrowableTest(unittest.TestCase):
    def test_negative_growth_rate(self):
        self.assertRaises(ValueError, lambda: Growable(np.zeros(10), 0, -1))

    def test_initialized_with_zero_elements(self):
        g = Growable(np.zeros(0), 0, 1)
        g.append(1)
        self.assertEqual(1, g.physical_size)
        self.assertEqual(1, g.logical_size)
        self.assertEqual(1, g._data[0])

    def test_logical_data(self):
        g = Growable(np.zeros(10), 0, 1)
        self.assertEqual(0, g.logical_data.shape[0])
        g.extend([1, 2])
        self.assertEqual(2, g.logical_data.shape[0])
        self.assertTrue(np.all(np.array([1, 2]) == g.logical_data))

    def test_append_no_growth(self):
        g = Growable(np.zeros(10), 0, 1)
        g.append(1)
        self.assertEqual(1, g._data[0])
        self.assertEqual(10, g.physical_size)
        self.assertEqual(1, g.logical_size)

    def test_extend_no_growth(self):
        g = Growable(np.zeros(10), 0, 1)
        g.extend(np.ones(3))
        self.assertTrue(np.all(1 == g._data[:3]))
        self.assertEqual(10, g.physical_size)
        self.assertEqual(3, g.logical_size)

    def test_append_growth(self):
        g = Growable(np.zeros(10), 10, 1)
        g.append(1)
        self.assertEqual(20, g.physical_size)
        self.assertEqual(11, g.logical_size)
        self.assertEqual(1, g._data[10])

    def test_extend_growth(self):
        g = Growable(np.zeros(10), 10, 1)
        g.extend(np.ones(5))
        self.assertEqual(20, g.physical_size)
        self.assertEqual(15, g.logical_size)
        self.assertTrue(np.all(1 == g._data[10:15]))

    def test_lt_one_growth_rate_append(self):
        g = Growable(np.zeros(1), 0, .00001)
        g.append(1)
        g.append(2)
        self.assertEqual(2, g.physical_size)
        self.assertEqual(2, g.logical_size)
        self.assertEqual(1, g._data[0])
        self.assertEqual(2, g._data[1])

    def test_multiple_grow_calls_required(self):
        g = Growable(np.zeros(10), 10, .1)
        g.extend(np.ones(4))
        self.assertEqual(14, g.physical_size)
        self.assertEqual(14, g.logical_size)
        self.assertTrue(np.all(1 == g._data[10:14]))

    def test_multidimensional(self):
        g = Growable(np.zeros((10, 3)), 10, 1)
        g.append(1)
        self.assertEqual(20, g.physical_size)
        self.assertEqual(11, g.logical_size)
        self.assertEqual((20, 3), g._data.shape)


class DownsampleTest(unittest.TestCase):
    def test_downsample_single_example_1D(self):
        a = np.ones(10)
        ds = downsample(a, 2)
        self.assertEqual((5,), ds.shape)

    def test_downsample_single_example_2D(self):
        a = np.ones((11, 11))
        ds = downsample(a, 2)
        self.assertEqual((5, 5), ds.shape)

    def test_downsample_multi_example_1D(self):
        a = np.ones((31, 10))
        ds = downsample(a, (2,))
        self.assertEqual((31, 5), ds.shape)

    def test_downsample_multi_example_2D(self):
        a = np.ones((31, 11, 11))
        ds = downsample(a, (2, 2))
        self.assertEqual((31, 5, 5), ds.shape)


class SlidingWindowTest(unittest.TestCase):
    def test_mismatched_dims_ws(self):
        a = np.zeros(10)
        self.assertRaises(ValueError, lambda: sliding_window(a, (1, 2)))

    def test_mismatched_dims_ss(self):
        a = np.zeros(10)
        self.assertRaises(ValueError, lambda: sliding_window(a, 3, (1, 2)))

    def test_windowsize_too_large_1D(self):
        a = np.zeros(10)
        self.assertRaises(ValueError, lambda: sliding_window(a, 11))

    def test_windowsize_too_large_2D(self):
        a = np.zeros((10, 10))
        self.assertRaises(ValueError, lambda: sliding_window(a, (3, 11)))

    def test_1D_no_step_specified(self):
        a = np.arange(10)
        b = sliding_window(a, 3)
        self.assertEqual((3, 3), b.shape)
        self.assertTrue(np.all(b.ravel() == a[:9]))

    def test_1D_with_step(self):
        a = np.arange(10)
        b = sliding_window(a, 3, 1)
        self.assertEqual((8, 3), b.shape)

    def test_1D_flat_nonflat_equivalent(self):
        a = np.zeros(10)
        bflat = sliding_window(a, 3)
        bnonflat = sliding_window(a, 3, flatten=False)
        self.assertEqual(bflat.shape, bnonflat.shape)
        self.assertTrue(np.all(bflat == bnonflat))

    def test_2D_no_step_specified(self):
        a = np.arange(64).reshape((8, 8))
        b = sliding_window(a, (4, 4))
        self.assertEqual((4, 4, 4), b.shape)

    def test_2D_with_step(self):
        a = np.zeros((8, 8))
        b = sliding_window(a, (4, 4), (1, 1))
        self.assertEqual((25, 4, 4), b.shape)

    def test_2D_nonflat_no_step_specified(self):
        a = np.arange(64).reshape((8, 8))
        b = sliding_window(a, (4, 4), flatten=False)
        self.assertEqual((2, 2, 4, 4), b.shape)

    def test_2D_nonflat_with_step(self):
        a = np.zeros((8, 8))
        b = sliding_window(a, (4, 4), (1, 1), flatten=False)
        self.assertEqual((5, 5, 4, 4), b.shape)

        # def test_reshape_does_not_create_copy(self):
        #     a = np.zeros((8,8))
        #     b = sliding_window(a,(2,2))
        #     b[:] = 1
        #     self.assertTrue(np.all(a == 1))
        #
        # def test_can_unwind(self):
        #     a = np.arange(64).reshape((8,8))
        #     b = sliding_window(a,(2,2))
        #     c = np.zeros((8,8))
        #     d = sliding_window(c,(2,2))
        #     d[:] = b
        #     self.assertTrue(np.all(a == c))


class WindowedTest(unittest.TestCase):
    def test_windowsize_ltone(self):
        a = np.arange(10)
        self.assertRaises(ValueError, lambda: windowed(a, 0, 1))

    def test_stepsize_ltone(self):
        a = np.arange(10)
        self.assertRaises(ValueError, lambda: windowed(a, 1, 0))

    def test_no_windowing(self):
        a = np.arange(10)
        l, w = windowed(a, 1, 1)
        self.assertTrue(a is w)
        self.assertEqual(0, l.shape[0])

    def test_drop_samples(self):
        a = np.arange(10)
        l, w = windowed(a, 1, 2)
        self.assertEqual(5, w.shape[0])
        self.assertEqual(0, l.shape[0])

    def test_windowsize_two_stepsize_one_cut(self):
        a = np.arange(10)
        l, w = windowed(a, 2, 1)
        self.assertEqual(1, l.shape[0])
        self.assertEqual((9, 2), w.shape)

    def test_windowsize_two_stepsize_one_pad(self):
        a = np.arange(10)
        l, w = windowed(a, 2, 1, True)
        self.assertEqual(0, l.shape[0])
        self.assertEqual((10, 2), w.shape)

    def test_windowsize_two_stepsize_two_cut(self):
        a = np.arange(10)
        l, w = windowed(a, 2, 2)
        self.assertEqual(0, l.shape[0])
        self.assertEqual((5, 2), w.shape)

    def test_windowsize_two_stepsize_two_pad(self):
        a = np.arange(10)
        l, w = windowed(a, 2, 2, True)
        self.assertEqual(0, l.shape[0])
        self.assertEqual((5, 2), w.shape)

    def test_windowsize_three_stepsize_two_cut(self):
        a = np.arange(10)
        l, w = windowed(a, 3, 2)
        self.assertEqual(2, l.shape[0])
        self.assertEqual((4, 3), w.shape)

    def test_windowsize_three_stepsize_two_pad(self):
        a = np.arange(10)
        l, w = windowed(a, 3, 2, dopad=True)
        self.assertEqual(0, l.shape[0])
        self.assertEqual((5, 3), w.shape)
        self.assertTrue(np.all([0, 0] == w[-1, -1]))

    def test_windowsize_three_stepsize_three_cut(self):
        a = np.arange(10)
        l, w = windowed(a, 3, 3)
        self.assertEqual(1, l.shape[0])
        self.assertEqual((3, 3), w.shape)

    def test_windowsize_three_stepsize_three_pad(self):
        a = np.arange(10)
        l, w = windowed(a, 3, 3, dopad=True)
        self.assertEqual(0, l.shape[0])
        self.assertEqual((4, 3), w.shape)

    def test_windowsize_gt_length_cut(self):
        a = np.arange(5)
        l, w = windowed(a, 6, 1)
        self.assertEqual(5, l.shape[0])
        self.assertEqual(0, w.shape[0])

    def test_windowsize_gt_length_pad(self):
        a = np.arange(5)
        l, w = windowed(a, 6, 1, dopad=True)
        self.assertEqual(0, l.shape[0])
        self.assertEqual((1, 6), w.shape)

    def test_twod_cut(self):
        a = np.arange(20).reshape((10, 2))
        l, w = windowed(a, 3, 2)
        self.assertEqual(2, l.shape[0])
        self.assertEqual((4, 3, 2), w.shape)

    def test_twod_pad(self):
        a = np.arange(20).reshape((10, 2))
        l, w = windowed(a, 3, 2, dopad=True)
        self.assertEqual(0, l.shape[0])
        self.assertEqual((5, 3, 2), w.shape)

    def test_no_stepsize_specified(self):
        a = np.arange(10)
        l, w = windowed(a, 2)
        self.assertEqual(0, l.shape[0])
        self.assertEqual((5, 2), w.shape)

    def test_can_apply_windowed_twice(self):
        a = np.arange(100)
        _, w = windowed(a, 7, 3)
        _, w2 = windowed(w, 3, 1)
        self.assertEqual((3, 7), w2.shape[1:])

    def test_can_apply_windowed_twice_2(self):
        samples = np.random.random_sample(44100)
        _, w = windowed(samples, 512, 256)
        f = np.fft.fft(w)[:, 1:]
        _, w2 = windowed(f, 3, 1)
        self.assertEqual((3, 511), w2.shape[1:])
