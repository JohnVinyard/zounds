from trainer import Trainer
from random import choice
import numpy as np


class TripletEmbeddingTrainer(Trainer):
    """
    Learn an embedding by applying the triplet loss to anchor examples, negative
    examples, and deformed or adjacent examples, akin to:

    * `UNSUPERVISED LEARNING OF SEMANTIC AUDIO REPRESENTATIONS` <https://arxiv.org/pdf/1711.02209.pdf>

    Args:
        network (nn.Module): the neural network to train
        epochs (int): the desired number of passes over the entire dataset
        batch_size (int): the number of examples in each minibatch
        anchor_slice (slice): since choosing examples near the anchor example
            is one possible transformation that can be applied to find a positive
            example, batches generally consist of examples that are longer
            (temporally) than the examples that will be fed to the network, so
            that adjacent examples may be chosen.  This slice indicates which
            part of the minibatch examples comprises the anchor
        deformations (callable): a collection of other deformations or
            transformations that can be applied to anchor examples to derive
            positive examples.  These callables should take two arguments: the
            anchor examples from the minibatch, as well as the "wider" minibatch
            examples that include temporally adjacent events
    """

    def __init__(
            self,
            network,
            epochs,
            batch_size,
            anchor_slice,
            deformations=None):

        super(TripletEmbeddingTrainer, self).__init__(epochs, batch_size)
        self.anchor_slice = anchor_slice
        self.network = network
        self.deformations = deformations

        # The margin hyperparameter is set to 0.1 in, according to section 4.2
        # of the paper https://arxiv.org/pdf/1711.02209.pdf
        self.margin = 0.1

    def _driver(self, data):
        batches_in_epoch = len(data) // self.batch_size
        for epoch in xrange(self.epochs):
            for batch in xrange(batches_in_epoch):
                yield epoch, batch

    def _apply_network_and_normalize(self, x):
        """
        Pass x through the network, and give the output unit norm, as specified
        by section 4.2 of https://arxiv.org/pdf/1711.02209.pdf
        """

        import torch
        x = self.network(x)
        return x / torch.norm(x, dim=1).view(-1, 1)

    def _select_batch(self, training_set):
        indices = np.random.randint(0, len(training_set), self.batch_size)
        batch = training_set[indices, self.anchor_slice]
        return indices, batch.astype(np.float32)

    def train(self, data):
        from torch import nn
        from torch.optim import Adam
        from util import to_var

        data = data['data']

        self.network.cuda()
        self.network.train()

        optimizer = Adam(self.network.parameters(), lr=1e-5)
        loss = nn.TripletMarginLoss(margin=self.margin).cuda()

        for epoch, batch in self._driver(data):
            self.network.zero_grad()

            # choose a batch of anchors
            indices, anchor = self._select_batch(data)
            anchor_v = to_var(anchor)
            a = self._apply_network_and_normalize(anchor_v)

            # choose negative examples
            negative_indices, negative = self._select_batch(data)
            negative_v = to_var(negative)
            n = self._apply_network_and_normalize(negative_v)

            # choose a deformation for this batch and apply it to produce the
            # positive examples
            deformation = choice(self.deformations)
            positive = deformation(anchor, data[indices, ...]) \
                .astype(np.float32)
            positive_v = to_var(positive)
            p = self._apply_network_and_normalize(positive_v)

            error = loss.forward(a, p, n)
            error.backward()
            optimizer.step()

            print epoch, batch, error.data.cpu().numpy().squeeze(), deformation

        return self.network
