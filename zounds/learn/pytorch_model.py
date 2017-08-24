from preprocess import Preprocessor, PreprocessResult, Op
import numpy as np


def train_autoencoder(model, data, loss, optimizer, epochs, batch_size):
    import torch
    from torch.autograd import Variable

    ae = model
    ae.cuda()
    loss.cuda()

    data = data.astype(np.float32)

    for epoch in xrange(epochs):
        for i in xrange(0, len(data), batch_size):

            ae.zero_grad()

            minibatch = data[i: i + batch_size]
            inp = torch.from_numpy(minibatch)
            inp = inp.cuda()
            inp_v = Variable(inp)
            output = ae(inp_v)

            error = loss(output, inp_v)
            error.backward()
            optimizer.step()
            e = error.data[0]

            if i % 10 == 0:
                print 'Epoch {epoch}, batch {i}, error {e}'.format(**locals())

    return ae


class PyTorchPreprocessResult(PreprocessResult):
    def __init__(self, data, op, inversion_data=None, inverse=None, name=None):
        super(PyTorchPreprocessResult, self).__init__(
            data, op, inversion_data, inverse, name)

    def __getstate__(self):
        forward_func = self.op._func
        inv_data_func = self.inversion_data._func
        backward_func = self.inverse._func
        autoencoder = self.op.autoencoder.state_dict()
        weights = dict(
            ((k, v.cpu().numpy()) for k, v in autoencoder.iteritems()))
        cls = self.op.autoencoder.__class__
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

        autoencoder = state['cls']()
        autoencoder.load_state_dict(restored_weights)
        autoencoder.cuda()
        self.op = Op(state['forward_func'], autoencoder=autoencoder)
        self.inversion_data = Op(state['inv_data_func'],
                                 autoencoder=autoencoder)
        self.inverse = Op(state['backward_func'])
        self.name = state['name']

    def for_storage(self):
        return PyTorchPreprocessResult(
            None,
            self.op,
            self.inversion_data,
            self.inverse,
            self.name)


class PyTorchAutoEncoder(Preprocessor):
    def __init__(
            self,
            model=None,
            loss=None,
            optimizer_func=None,
            epochs=None,
            batch_size=None,
            needs=None):
        super(PyTorchAutoEncoder, self).__init__(needs=needs)
        self.optimizer_func = optimizer_func
        self.epochs = epochs
        self.batch_size = batch_size
        self.loss = loss
        self.model = model

    def _forward_func(self):
        def x(d, autoencoder=None):
            import torch
            from torch.autograd import Variable
            import numpy as np
            tensor = torch.from_numpy(d.astype(np.float32))
            gpu = tensor.cuda()
            v = Variable(gpu)
            return autoencoder.encoder(v).data.cpu().numpy()

        return x

    def _backward_func(self):
        def x(d, autoencoder=None):
            import torch
            from torch.autograd import Variable
            tensor = torch.from_numpy(d)
            gpu = tensor.cuda()
            v = Variable(gpu)
            return autoencoder.decoder(v).data.cpu().numpy()

        return x

    def _process(self, data):
        data = self._extract_data(data)

        trained_autoencoder = train_autoencoder(
            self.model,
            data,
            self.loss,
            self.optimizer_func(self.model),
            self.epochs,
            self.batch_size)

        ff = self._forward_func()
        processed_data = ff(data, autoencoder=trained_autoencoder)
        op = self.transform(autoencoder=trained_autoencoder)
        inv_data = self.inversion_data(autoencoder=trained_autoencoder)
        inv = self.inverse_transform()

        yield PyTorchPreprocessResult(
            processed_data,
            op,
            inversion_data=inv_data,
            inverse=inv,
            name='PyTorchAutoEncoder')
