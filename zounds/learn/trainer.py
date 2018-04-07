import numpy as np


class Trainer(object):
    def __init__(self, epochs, batch_size, checkpoint_epochs=1):
        super(Trainer, self).__init__()
        self.checkpoint_epochs = checkpoint_epochs
        self.batch_size = batch_size
        self.epochs = epochs
        self._batch_complete_callbacks = dict()
        self._current_epoch = 0
        self.use_cuda = False

    def _cuda(self, device=None):
        pass

    def cuda(self, device=None):
        self.use_cuda = True
        self.network = self.network.cuda(device=device)
        self._cuda(device=device)
        return self

    def _variable(self, x, *args, **kwargs):
        from torch.autograd import Variable
        v = Variable(x, *args, **kwargs)
        if self.use_cuda:
            v = v.cuda()
        return v

    def _tensor(self, shape):
        import torch
        ft = torch.cuda.FloatTensor if self.use_cuda else torch.FloatTensor
        return ft(*shape)

    def _log(self, *args, **kwargs):
        if kwargs['batch'] % 10:
            return
        print kwargs

    def _zero_grad(self):
        self.network.zero_grad()

    def _minibatch(self, data):
        import torch
        indices = np.random.randint(0, len(data), self.batch_size)
        batch = torch.from_numpy(data[indices, ...])
        return self._variable(batch)

    def _training_step(self, epoch, batch, data):
        raise NotImplementedError()

    def train(self, data):
        start = self._current_epoch
        stop = self._current_epoch + self.checkpoint_epochs

        for epoch in xrange(start, stop):
            if epoch > self.epochs:
                break

            for batch in xrange(0, len(data), self.batch_size):
                results = self._training_step(epoch, batch, data)
                results.update(epoch=epoch, batch=batch, network=self.network)
                self.on_batch_complete(**results)

            self._current_epoch += 1

        return self.network

    def register_batch_complete_callback(self, callback):
        self._batch_complete_callbacks[id(callback)] = callback

    def unregister_batch_complete_callback(self, callback):
        try:
            del self._batch_complete_callbacks[id(callback)]
        except KeyError:
            # the callback was never registered
            pass

    def on_batch_complete(self, *args, **kwargs):
        for callback in self._batch_complete_callbacks.itervalues():
            callback(*args, **kwargs)
