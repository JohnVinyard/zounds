from hashlib import md5
import warnings
import featureflow as ff
from preprocess import Preprocessor, PreprocessResult, Op


class PyTorchPreprocessResult(PreprocessResult):
    # network_cache = dict()

    def __init__(self, data, op, inversion_data=None, inverse=None, name=None):
        super(PyTorchPreprocessResult, self).__init__(
            data, op, inversion_data, inverse, name)

    def __getstate__(self):
        forward_func = self.op._func
        inv_data_func = self.inversion_data._func
        backward_func = self.inverse._func
        network_params = self.op.network.state_dict()
        weights = dict(
            ((k, v.cpu().numpy()) for k, v in network_params.iteritems()))
        cls = self.op.network.__class__
        name = self.name

        kwargs = dict(self.op.kwargs)
        del kwargs['network']

        return dict(
            forward_func=forward_func,
            op_kwargs=kwargs,
            inv_data_func=inv_data_func,
            backward_func=backward_func,
            weights=weights,
            name=name,
            cls=cls)

    # def _network_identifier(self, weight_dict):
    #     sorted_keys = sorted(weight_dict.iterkeys())
    #     hashed = md5()
    #     for key in sorted_keys:
    #         value = weight_dict[key]
    #         hashed.update(key)
    #         hashed.update(value)
    #     return hashed.hexdigest()

    def __setstate__(self, state):
        import torch

        restored_weights = dict(
            ((k, torch.from_numpy(v).cuda())
             for k, v in state['weights'].iteritems()))
        network = state['cls']()
        network.load_state_dict(restored_weights)
        network.cuda()
        network.eval()

        self.op = Op(
            state['forward_func'], network=network, **state['op_kwargs'])
        self.inversion_data = Op(state['inv_data_func'],
                                 network=network)
        self.inverse = Op(state['backward_func'])
        self.name = state['name']

    def for_storage(self):
        return PyTorchPreprocessResult(
            None,
            self.op,
            self.inversion_data,
            self.inverse,
            self.name)


class PyTorchNetwork(Preprocessor):
    def __init__(self, trainer=None, post_training_func=None, needs=None):

        super(PyTorchNetwork, self).__init__(needs=needs)
        self.trainer = trainer
        self.post_training_func = post_training_func or (lambda x: x)
        self._cache = dict()

    def _forward_func(self):
        def x(d, network=None):
            from zounds.core import ArrayWithUnits, IdentityDimension
            from zounds.learn import apply_network

            result = apply_network(network, d, chunksize=128)
            try:
                return ArrayWithUnits(
                    result, [d.dimensions[0], IdentityDimension()])
            except AttributeError:
                return result
            except ValueError:
                # the number of dimensions has likely changed
                return result

        return x

    def _backward_func(self):
        def x(_):
            raise NotImplementedError()

        return x

    def _enqueue(self, data, pusher):
        k = self._dependency_name(pusher)
        self._cache[k] = data

    def _dequeue(self):
        if not self._finalized:
            raise ff.NotEnoughData()
        return self._cache

    def _process(self, data):
        data = self._extract_data(data)

        trained_network = self.trainer.train(data)

        try:
            forward_func = self._forward_func()
            x = self.post_training_func(data['data'])
            processed_data = forward_func(x, network=trained_network)
        except RuntimeError as e:
            processed_data = None
            # the dataset may be too large to fit onto the GPU all at once
            warnings.warn(e.message)

        op = self.transform(network=trained_network)
        inv_data = self.inversion_data()
        inv = self.inverse_transform()

        yield PyTorchPreprocessResult(
            processed_data,
            op,
            inversion_data=inv_data,
            inverse=inv,
            name='PyTorchNetwork')


class PyTorchGan(PyTorchNetwork):
    def __init__(self, apply_network='generator', trainer=None, needs=None):
        super(PyTorchGan, self).__init__(trainer=trainer, needs=needs)

        if apply_network not in ('generator', 'discriminator'):
            raise ValueError(
                'apply_network must be one of (generator, discriminator)')

        self.apply_network = apply_network
        self._cache = None

    def _forward_func(self):
        def x(d, network=None, apply_network=None):
            import torch
            from torch.autograd import Variable
            import numpy as np
            from zounds.core import ArrayWithUnits, IdentityDimension

            if apply_network == 'generator':
                n = network.generator
            else:
                n = network.discriminator

            chunks = []
            batch_size = 128

            for i in xrange(0, len(d), batch_size):
                tensor = torch.from_numpy(
                    d[i:i + batch_size].astype(np.float32))
                gpu = tensor.cuda()
                v = Variable(gpu)
                chunks.append(n(v).data.cpu().numpy())

            result = np.concatenate(chunks)

            try:
                return ArrayWithUnits(
                    result, d.dimensions[:-1] + (IdentityDimension(),))
            except AttributeError:
                return result
            except ValueError:
                # the number of dimensions has likely changed
                return result

        return x

    def _backward_func(self):
        def x(_):
            raise NotImplementedError()

        return x

    def _enqueue(self, data, pusher):
        self._cache = data

    def _process(self, data):
        data = self._extract_data(data)

        network = self.trainer.train(data)
        network.eval()

        try:
            # note that the processed data passed on to the next step in the
            # training pipeline will be the labels output by the discriminator
            forward_func = self._forward_func()
            processed_data = forward_func(
                data, network=network, apply_network='discriminator')
        except RuntimeError as e:
            processed_data = None
            # the dataset may be too large to fit onto the GPU all at once
            warnings.warn(e.message)

        op = self.transform(network=network, apply_network=self.apply_network)
        print op.network
        print op.apply_network
        inv_data = self.inversion_data()
        inv = self.inverse_transform()

        yield PyTorchPreprocessResult(
            processed_data,
            op,
            inversion_data=inv_data,
            inverse=inv,
            name='PyTorchGan')


class PyTorchAutoEncoder(PyTorchNetwork):
    def __init__(self, trainer=None, needs=None):
        super(PyTorchAutoEncoder, self).__init__(trainer=trainer, needs=needs)
        self._cache = None

    def _forward_func(self):
        def x(d, network=None):
            import torch
            from torch.autograd import Variable
            import numpy as np
            from zounds.core import ArrayWithUnits, IdentityDimension

            chunks = []
            batch_size = 128

            for i in xrange(0, len(d), batch_size):
                tensor = torch.from_numpy(
                    d[i:i + batch_size].astype(np.float32))
                gpu = tensor.cuda()
                v = Variable(gpu)
                chunks.append(network.encoder(v).data.cpu().numpy())

            encoded = np.concatenate(chunks)

            try:
                extra_dims = (IdentityDimension(),) * (encoded.ndim - 1)
                return ArrayWithUnits(
                    encoded, d.dimensions[:1] + extra_dims)
            except AttributeError:
                return encoded

        return x

    def _backward_func(self):
        def x(d, network=None):
            import torch
            from torch.autograd import Variable
            import numpy as np

            chunks = []
            batch_size = 128

            for i in xrange(0, len(d), batch_size):
                tensor = torch.from_numpy(d.astype(np.float32))
                gpu = tensor.cuda()
                v = Variable(gpu)
                chunks.append(network.decoder(v).data.cpu().numpy())

            decoded = np.concatenate(chunks)
            return decoded

        return x

    def _enqueue(self, data, pusher):
        self._cache = data

    def _process(self, data):
        data = self._extract_data(data)

        data = dict(data=data, labels=data)

        trained_network = self.trainer.train(data)
        trained_network.eval()

        processed_data = None
        inp = data['data']

        while processed_data is None:
            try:
                forward_func = self._forward_func()
                processed_data = forward_func(inp, network=trained_network)
            except RuntimeError as e:
                processed_data = None
                warnings.warn(e.message)
                # we've just experienced an out of memory exception.  Cut the
                # size of the input data in half, so that downstream nodes that
                # need some data to initialize themselves can do so
                inp = inp[:len(inp) // 64]
            except ValueError:
                break

        op = self.transform(network=trained_network)
        inv_data = self.inversion_data(network=trained_network)
        inv = self.inverse_transform()

        yield PyTorchPreprocessResult(
            processed_data,
            op,
            inversion_data=inv_data,
            inverse=inv,
            name='PyTorchAutoEncoder')
