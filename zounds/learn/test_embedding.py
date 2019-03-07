import torch
import unittest2
from .embedding import TripletEmbeddingTrainer
from torch import nn
import numpy as np


class TripletEmbeddingTrainerTests(unittest2.TestCase):
    def test_normalization_does_not_cause_nans(self):
        class Network(nn.Module):
            def __init__(self):
                super(Network, self).__init__()

            def forward(self, x):
                return x

        network = Network()
        trainer = TripletEmbeddingTrainer(network, 100, 32, slice(None))
        x = torch.zeros(8, 3)
        result = trainer._apply_network_and_normalize(x).data.numpy()
        self.assertFalse(np.any(np.isnan(result)))
