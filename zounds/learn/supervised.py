from trainer import Trainer
import numpy as np


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
        self.model = model
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

    def train(self, data):
        import torch
        from torch.autograd import Variable

        model = self.model.cuda()
        loss = self.loss.cuda()

        data, labels = data['data'], data['labels']

        test_size = int(self.holdout_percent * len(data))
        test_data, test_labels = data[:test_size], labels[:test_size]
        data, labels = data[test_size:], labels[test_size:]

        def batch(d, l, test=False):
            d = self.data_preprocessor(d)
            l = self.label_preprocessor(l)
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

            self.samples = zip(inp_v, output)
            return inp_v, output, error.data[0]

        start = self._current_epoch
        stop = self._current_epoch + self.checkpoint_epochs

        for epoch in xrange(start, stop):

            if epoch >= self.epochs:
                break

            for i in xrange(0, len(data), self.batch_size):

                model.zero_grad()

                # training batch
                minibatch_slice = slice(i, i + self.batch_size)
                minibatch_data = data[minibatch_slice]
                minibatch_labels = labels[minibatch_slice]

                inp, output, e = batch(
                    minibatch_data, minibatch_labels, test=False)

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

        return model
