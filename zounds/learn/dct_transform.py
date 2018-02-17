from zounds.spectral import dct_basis
import torch
from torch.autograd import Variable


class DctTransform(object):
    def __init__(self, use_cuda=False):
        super(DctTransform, self).__init__()
        self.use_cuda = use_cuda
        self._basis_cache = dict()
        self._window_cache = dict()

    def _variable(self, x, *args, **kwargs):
        v = Variable(x, *args, **kwargs)
        if self.use_cuda:
            v = v.cuda()
        return v

    def dct_basis(self, n):
        try:
            return self._basis_cache[n]
        except KeyError:
            basis = torch.from_numpy(dct_basis(n)).float()
            if self.use_cuda:
                basis = basis.cuda()
            self._basis_cache[n] = basis
            return basis

    def window(self, n, window):
        try:
            return self._window_cache[n]
        except KeyError:
            data = torch.from_numpy(window._wdata(n)).float()
            if self.use_cuda:
                data = data.cuda()
            self._window_cache[n] = data
            return data

    def _base_dct_transform(self, x, basis, axis=-1):
        n = torch.FloatTensor(1)
        n[:] = 2. / x.shape[axis]
        n = self._variable(n)
        coeffs = torch.matmul(x, basis) * torch.sqrt(n)
        return coeffs

    def dct(self, x, axis=-1):
        basis = self._variable(self.dct_basis(x.shape[axis]))
        return self._base_dct_transform(x, basis, axis)

    def idct(self, x, axis=-1):
        basis = self._variable(self.dct_basis(x.shape[axis]))
        return self._base_dct_transform(x, basis.t(), axis)

    def dct_resample(self, x, factor, axis=-1):

        # figure out how many samples our resampled signal will have
        n_samples = int(factor * x.shape[axis])

        coeffs = self.dct(x)

        # create the shape of our new coefficients
        new_coeffs_shape = list(coeffs.shape)
        new_coeffs_shape[axis] = n_samples

        # fill in the new, resampled coefficients
        new_coeffs = self._variable(torch.zeros(*new_coeffs_shape))

        new_coeffs_slices = [slice(None)] * x.dim()
        new_coeffs_slices[axis] = slice(None, coeffs.shape[axis])
        new_coeffs_slices = tuple(new_coeffs_slices)

        old_coeffs_slices = [slice(None)] * x.dim()
        old_coeffs_slices[axis] = slice(None, new_coeffs.shape[axis])
        old_coeffs_slices = tuple(old_coeffs_slices)

        new_coeffs[new_coeffs_slices] = coeffs[old_coeffs_slices]

        return self.idct(new_coeffs)

    def short_time_dct(self, x, size, step, window):
        original_shape = x.shape
        x = x.unfold(-1, size, step)
        window = self._variable(self.window(x.shape[-1], window))
        x = x * window
        x = self.dct(x, axis=-1)
        x = x.view((original_shape[0], size, x.shape[2]))
        return x
