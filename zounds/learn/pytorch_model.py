import warnings
import featureflow as ff
from preprocess import Preprocessor, PreprocessResult, Op
import numpy as np


class Trainer(object):
    def __init__(self, epochs, batch_size):
        super(Trainer, self).__init__()
        self.batch_size = batch_size
        self.epochs = epochs

    def train(self):
        raise NotImplemented()


class GanTrainer(Trainer):
    def __init__(
            self,
            generator,
            discriminator,
            loss,
            generator_optim_func,
            discriminator_optim_func,
            latent_dimension,
            epochs,
            batch_size):

        super(GanTrainer, self).__init__(epochs, batch_size)
        self.discriminator_optim_func = discriminator_optim_func
        self.generator_optim_func = generator_optim_func
        self.latent_dimension = latent_dimension
        self.loss = loss
        self.discriminator = discriminator
        self.generator = generator

    def train(self, data):

        import torch
        from torch.autograd import Variable

        data = data.astype(np.float32)

        zdim = self.latent_dimension

        # TODO: These dimensions work for vanilla GANs, but need to be
        # reversed (batch_size, zdim, 1) for convolutional GANs

        noise_shape = (self.batch_size,) + self.latent_dimension
        noise = torch.FloatTensor(*noise_shape)
        fixed_noise = torch.FloatTensor(*noise_shape).normal_(0, 1)
        label = torch.FloatTensor(self.batch_size)
        real_label = 1
        fake_label = 0

        self.generator.cuda()
        self.discriminator.cuda()
        self.loss.cuda()
        label = label.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        generator_optim = self.generator_optim_func(self.generator)
        discriminator_optim = self.discriminator_optim_func(self.discriminator)

        for epoch in xrange(self.epochs):
            for i in xrange(0, len(data), self.batch_size):
                minibatch = data[i: i + self.batch_size]

                if len(minibatch) != self.batch_size:
                    continue

                inp = torch.from_numpy(minibatch)
                inp = inp.cuda()

                # train discriminator on real data with one-sided label
                # smoothing
                self.discriminator.zero_grad()
                soft_labels = \
                    0.7 + (np.random.random_sample(self.batch_size) * 0.4) \
                        .astype(np.float32)
                soft_labels = torch.from_numpy(soft_labels)
                label.copy_(soft_labels)

                input_v = Variable(inp)
                label_v = Variable(label)

                output = self.discriminator.forward(input_v)
                real_error = self.loss(output, label_v)
                real_error.backward()

                # train discriminator on fake data
                noise.normal_(0, 1)
                noise_v = Variable(noise)
                fake = self.generator.forward(noise_v)
                label.fill_(fake_label)
                label_v = Variable(label)
                output = self.discriminator.forward(fake.detach())
                fake_error = self.loss(output, label_v)
                fake_error.backward()
                discriminator_error = real_error + fake_error
                discriminator_optim.step()

                # train generator
                self.generator.zero_grad()
                label.fill_(real_label)
                label_v = Variable(label)
                output = self.discriminator.forward(fake)
                generator_error = self.loss(output, label_v)
                generator_error.backward()
                generator_optim.step()

                gl = generator_error.data[0]
                dl = discriminator_error.data[0]
                re = real_error.data[0]
                fe = fake_error.data[0]

                if i % 10 == 0:
                    print \
                        'Epoch {epoch}, batch {i}, generator {gl}, real_error {re}, fake_error {fe}' \
                            .format(**locals())

        return self.generator, self.discriminator


class SupervisedTrainer(Trainer):
    def __init__(
            self,
            model,
            loss,
            optimizer,
            epochs,
            batch_size,
            holdout_percent=0.0):

        super(SupervisedTrainer, self).__init__(
            epochs,
            batch_size)

        self.holdout_percent = holdout_percent
        self.optimizer = optimizer(model)
        self.loss = loss
        self.model = model

    def train(self, data):
        import torch
        from torch.autograd import Variable

        model = self.model.cuda()
        loss = self.loss.cuda()

        data, labels = data['data'], data['labels']

        test_size = int(self.holdout_percent * len(data))
        test_data, test_labels = data[:test_size], labels[:test_size]
        data, labels = data[test_size:], labels[test_size:]

        if data is labels:
            # this is an autoencoder scenario, so let's saved on memory
            data = data.astype(np.float32)
            test_data = data.astype(np.float32)
            labels = data
            test_labels = labels
        else:
            data = data.astype(np.float32)
            test_data = test_data.astype(np.float32)
            labels = labels.astype(np.float32)
            test_labels = test_labels.astype(np.float32)

        def batch(d, l, test=False):
            inp = torch.from_numpy(d)
            inp = inp.cuda()
            inp_v = Variable(inp, volatile=test)
            output = model(inp_v)

            labels_t = torch.from_numpy(l)
            labels_t = labels_t.cuda()
            labels_v = Variable(labels_t)

            error = loss(output, labels_v)

            if not test:
                error.backward()
                self.optimizer.step()

            return error.data[0]

        for epoch in xrange(self.epochs):
            for i in xrange(0, len(data), self.batch_size):

                model.zero_grad()

                # training batch
                minibatch_slice = slice(i, i + self.batch_size)
                minibatch_data = data[minibatch_slice]
                minibatch_labels = labels[minibatch_slice]

                e = training_error = batch(
                    minibatch_data, minibatch_labels, test=False)

                # test batch
                if test_size:
                    indices = np.random.randint(0, test_size, self.batch_size)
                    test_batch_data = test_data[indices, ...]
                    test_batch_labels = test_labels[indices, ...]

                    te = test_error = batch(
                        test_batch_data, test_batch_labels, test=True)
                else:
                    te = 'n/a'

                if i % 10 == 0:
                    print 'Epoch {epoch}, batch {i}, train error {e}, test error {te}'.format(
                        **locals())

        return model


class PyTorchPreprocessResult(PreprocessResult):
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
        return dict(
            forward_func=forward_func,
            inv_data_func=inv_data_func,
            backward_func=backward_func,
            weights=weights,
            name=name,
            cls=cls)

    def __setstate__(self, state):
        import torch
        restored_weights = dict(
            ((k, torch.from_numpy(v).cuda())
             for k, v in state['weights'].iteritems()))

        network = state['cls']()
        network.load_state_dict(restored_weights)
        network.cuda()
        self.op = Op(state['forward_func'], network=network)
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
    def __init__(self, trainer=None, needs=None):
        super(PyTorchNetwork, self).__init__(needs=needs)
        self.trainer = trainer
        self._cache = dict()

    def _forward_func(self):
        def x(d, network=None):
            import torch
            from torch.autograd import Variable
            import numpy as np
            from zounds.core import ArrayWithUnits, IdentityDimension
            tensor = torch.from_numpy(d.astype(np.float32))
            gpu = tensor.cuda()
            v = Variable(gpu)
            result = network(v).data.cpu().numpy()
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
            processed_data = forward_func(data['data'], network=trained_network)
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
    def __init__(self, trainer=None, needs=None):
        super(PyTorchGan, self).__init__(trainer=trainer, needs=needs)
        self._cache = None

    def _enqueue(self, data, pusher):
        self._cache = data

    def _process(self, data):
        data = self._extract_data(data)

        generator, discriminator = self.trainer.train(data)

        try:
            # note that the processed data passed on to the next step in the
            # training pipeline will be the labels output by the discriminator
            forward_func = self._forward_func()
            processed_data = forward_func(data, network=discriminator)
        except RuntimeError as e:
            processed_data = None
            # the dataset may be too large to fit onto the GPU all at once
            warnings.warn(e.message)

        op = self.transform(network=generator)
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
            tensor = torch.from_numpy(d.astype(np.float32))
            gpu = tensor.cuda()
            v = Variable(gpu)
            encoded = network.encoder(v).data.cpu().numpy()
            try:
                return ArrayWithUnits(
                    encoded, d.dimensions[:-1] + (IdentityDimension(),))
            except AttributeError:
                return encoded

        return x

    def _backward_func(self):
        def x(d, network=None):
            import torch
            from torch.autograd import Variable
            import numpy as np
            tensor = torch.from_numpy(d.astype(np.float32))
            gpu = tensor.cuda()
            v = Variable(gpu)
            return network.decoder(v).data.cpu().numpy()

        return x

    def _enqueue(self, data, pusher):
        self._cache = data

    def _process(self, data):
        data = self._extract_data(data)

        data = dict(data=data, labels=data)

        trained_network = self.trainer.train(data)

        try:
            forward_func = self._forward_func()
            processed_data = forward_func(data['data'], network=trained_network)
        except RuntimeError as e:
            processed_data = None
            warnings.warn(e.message)

        op = self.transform(network=trained_network)
        inv_data = self.inversion_data(network=trained_network)
        inv = self.inverse_transform()

        yield PyTorchPreprocessResult(
            processed_data,
            op,
            inversion_data=inv_data,
            inverse=inv,
            name='PyTorchAutoEncoder')