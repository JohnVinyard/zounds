from trainer import Trainer
import numpy as np
import warnings


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
            label_preprocessor=lambda x: x,
            checkpoint_epochs=1):

        super(SupervisedTrainer, self).__init__(
            epochs,
            batch_size,
            checkpoint_epochs=checkpoint_epochs)

        self.label_preprocessor = label_preprocessor
        self.data_preprocessor = data_preprocessor
        self.holdout_percent = holdout_percent
        self.optimizer = optimizer(model)
        self.loss = loss
        self.network = model
        self.register_batch_complete_callback(self._log)
        self.samples = None

    def _log(self, *args, **kwargs):
        if kwargs['batch'] % 10:
            return
        msg = \
            'Epoch {epoch}, batch {batch}, train error ' \
            '{train_error}, test error {test_error}'
        print msg.format(**kwargs)

    def random_sample(self):
        if self.samples is None:
            raise RuntimeError(
                'There are no samples yet.  Has training started?')
        index = np.random.randint(0, len(self.samples))
        inp, label = self.samples[index]
        return inp, label

    def _cuda(self, device=None):
        self.network = self.network.cuda()
        self.loss = self.loss.cuda()

    def train(self, data):
        data, labels = data['data'], data['labels']

        test_size = int(self.holdout_percent * len(data))
        test_data, test_labels = data[:test_size], labels[:test_size]
        data, labels = data[test_size:], labels[test_size:]

        def batch(d, l, test=False):
            d = self.data_preprocessor(d)
            l = self.label_preprocessor(l)
            inp_v = self._variable(d, volatile=test)
            output = self.network(inp_v)

            labels_v = self._variable(l)

            error = self.loss(output, labels_v)

            if not test:
                error.backward()
                self.optimizer.step()

            self.samples = zip(inp_v, output)
            return inp_v, output, error.data.item()

        start = self._current_epoch
        stop = self._current_epoch + self.checkpoint_epochs

        for epoch in xrange(start, stop):

            if epoch >= self.epochs:
                break

            for i in xrange(0, len(data), self.batch_size):

                self.network.zero_grad()

                # training batch
                minibatch_slice = slice(i, i + self.batch_size)
                minibatch_data = data[minibatch_slice]
                minibatch_labels = labels[minibatch_slice]

                try:
                    inp, output, e = batch(
                        minibatch_data, minibatch_labels, test=False)
                except RuntimeError as e:
                    if 'Assert' in e.message:
                        warnings.warn(e.message)
                        continue
                    else:
                        raise

                # test batch
                if test_size:
                    indices = np.random.randint(0, test_size, self.batch_size)
                    test_batch_data = test_data[indices, ...]
                    test_batch_labels = test_labels[indices, ...]

                    inp, output, te = batch(
                        test_batch_data, test_batch_labels, test=True)
                else:
                    te = 'n/a'

                self.on_batch_complete(
                    epoch=epoch,
                    batch=i,
                    train_error=te,
                    test_error=e,
                    samples=self.samples)

            self._current_epoch += 1

        return self.network
