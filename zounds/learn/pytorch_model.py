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


class WassersteinGanTrainer(Trainer):
    """
    Args:
        preprocess (function): function that takes the current epoch, and
        a minibatch, and mutates the minibatch
        arg_maker (function): function that takes the current epoch and outputs
            args to pass to the generator and discriminator
    """

    def __init__(
            self,
            network,
            latent_dimension,
            n_critic_iterations,
            epochs,
            batch_size,
            preprocess=None,
            arg_maker=None,
            generator_loss_term=lambda network, output: 0,
            critic_loss_term=lambda network, output: 0):

        super(WassersteinGanTrainer, self).__init__(epochs, batch_size)
        self.critic_loss_term = critic_loss_term
        self.generator_loss_term = generator_loss_term
        self.arg_maker = arg_maker
        self.preprocess = preprocess
        self.n_critic_iterations = n_critic_iterations
        self.latent_dimension = latent_dimension
        self.network = network
        self.critic = network.discriminator
        self.generator = network.generator

    def _minibatch(self, data):
        indices = np.random.randint(0, len(data), self.batch_size)
        return data[indices, ...]

    def _weight_initialization(self, layer):
        from torch import nn
        if isinstance(layer, nn.Conv1d) or isinstance(layer, nn.ConvTranspose1d):
            layer.weight.data.normal_(0, 0.02)
        elif isinstance(layer, nn.BatchNorm1d):
            layer.weight.data.normal_(1, 0.02)
            layer.bias.data.fill_(0)

    def _gradient_penalty(self, real_samples, fake_samples):
        """
        Compute the norm of the gradients for each sample in a batch, and
        penalize anything on either side of unit norm
        """
        import torch
        from torch.autograd import Variable, grad

        # computing the norm of the gradients is very expensive, so I'm only
        # taking a subset of the minibatch here
        subset_size = 10

        real_samples = real_samples[:subset_size]
        fake_samples = fake_samples[:subset_size]

        # TODO: this should have the same number of dimensions as real and
        # fake samples, and should not be hard-coded
        alpha = torch.rand(subset_size, 1).cuda()

        interpolates = alpha * real_samples + ((1 - alpha) * fake_samples)
        interpolates = Variable(interpolates.cuda(), requires_grad=True)

        d_output = self.critic(interpolates)

        gradients = grad(
            outputs=d_output,
            inputs=interpolates,
            grad_outputs=torch.ones(d_output.size()).cuda(),
            create_graph=True,
            retain_graph=True,
            only_inputs=True)[0]
        return ((gradients.norm(2, dim=1) - 1) ** 2).mean() * 10

    def train(self, data):

        import torch
        from torch.optim import Adam
        from torch.autograd import Variable

        data = data.astype(np.float32)

        zdim = self.latent_dimension

        noise_shape = (self.batch_size,) + self.latent_dimension
        noise = torch.FloatTensor(*noise_shape)
        fixed_noise = torch.FloatTensor(*noise_shape).normal_(0, 1)

        self.generator.cuda()
        self.critic.cuda()
        noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

        # self.generator.apply(self._weight_initialization)
        # self.critic.apply(self._weight_initialization)

        trainable_generator_params = (
            p for p in self.generator.parameters() if p.requires_grad)
        trainable_critic_params = (
            p for p in self.critic.parameters() if p.requires_grad)

        generator_optim = Adam(
            trainable_generator_params, lr=0.0001, betas=(0, 0.9))
        critic_optim = Adam(
            trainable_critic_params, lr=0.0001, betas=(0, 0.9))

        for epoch in xrange(self.epochs):
            if self.arg_maker:
                kwargs = self.arg_maker(epoch)
            else:
                kwargs = dict()

            for i in xrange(0, len(data), self.batch_size):
                self.generator.zero_grad()
                self.critic.zero_grad()

                self.generator.eval()
                self.critic.train()

                for c in xrange(self.n_critic_iterations):

                    self.critic.zero_grad()

                    minibatch = self._minibatch(data)
                    inp = torch.from_numpy(minibatch)
                    inp = inp.cuda()
                    input_v = Variable(inp)

                    if self.preprocess:
                        input_v = self.preprocess(epoch, input_v)

                    d_real = self.critic.forward(input_v, **kwargs)

                    # train discriminator on fake data
                    noise.normal_(0, 1)
                    noise_v = Variable(noise, volatile=True)
                    fake = Variable(
                        self.generator.forward(noise_v, **kwargs).data)

                    if self.preprocess:
                        fake = self.preprocess(epoch, fake)

                    d_fake = self.critic.forward(fake, **kwargs)

                    real_mean = torch.mean(d_real)
                    fake_mean = torch.mean(d_fake)
                    d_loss = \
                        (fake_mean - real_mean) \
                        + self._gradient_penalty(input_v.data, fake.data) \
                        + self.critic_loss_term(self.critic, d_fake)
                    d_loss.backward()
                    critic_optim.step()

                self.generator.train()
                self.critic.eval()

                # train generator
                noise.normal_(0, 1)
                noise_v = Variable(noise)
                fake = self.generator.forward(noise_v, **kwargs)

                if self.preprocess:
                    fake = self.preprocess(epoch, fake)

                d_fake = self.critic.forward(fake, **kwargs)
                g_loss = \
                    -torch.mean(d_fake) \
                    + self.generator_loss_term(self.generator, fake)
                g_loss.backward()
                generator_optim.step()

                gl = g_loss.data[0]
                dl = d_loss.data[0]

                if i % 10 == 0:
                    print \
                        'Epoch {epoch}, batch {i}, generator {gl}, critic {dl}' \
                            .format(**locals())

        return self.network


class GanTrainer(Trainer):
    def __init__(
            self,
            network,
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
        self.network = network
        self.discriminator = network.discriminator
        self.generator = network.generator

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

        return self.network


class SupervisedTrainer(Trainer):
    def __init__(
            self,
            model,
            loss,
            optimizer,
            epochs,
            batch_size,
            holdout_percent=0.0,
            data_preprocessor=lambda x: x,
            label_preprocessor=lambda x: x):

        super(SupervisedTrainer, self).__init__(
            epochs,
            batch_size)

        self.label_preprocessor = label_preprocessor
        self.data_preprocessor = data_preprocessor
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
            d = self.data_preprocessor(d).astype(np.float32)
            l = self.label_preprocessor(l).astype(np.float32)
            inp = torch.from_numpy(d)
            inp = inp.cuda()
            inp_v = Variable(inp, volatile=test)
            output = model(inp_v)

            labels_t = torch.from_numpy(l)
            labels_t = labels_t.cuda()
            labels_v = Variable(labels_t)

            error = loss(output, labels_v)

            if not test:
                grad = error.backward()
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

        kwargs = self.op.kwargs
        del kwargs['network']

        return dict(
            forward_func=forward_func,
            op_kwargs=kwargs,
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

            # tensor = torch.from_numpy(d.astype(np.float32))
            # gpu = tensor.cuda()
            # v = Variable(gpu)
            # result = n(v).data.cpu().numpy()

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
