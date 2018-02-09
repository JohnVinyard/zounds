import torch
from torch import nn
from torch.autograd import Variable
import torch.nn.functional as F


class RawSampleEmbedding(nn.Module):
    """
    Embed raw audio samples after quantizing them and applying a
    softmax/categorical distribution
    """

    def __init__(self, n_categories, embedding_dim):
        super(RawSampleEmbedding, self).__init__()
        self.n_categories = n_categories
        self.embedding_dim = embedding_dim
        self.linear = nn.Linear(
            self.n_categories, self.embedding_dim)

    def _mu_law(self, x):
        m = Variable(torch.FloatTensor(1))
        m[:] = self.n_categories + 1
        s = torch.sign(x)
        x = torch.abs(x)
        x = s * (torch.log(1 + (self.n_categories * x)) / torch.log(m))
        return x

    def _shift_and_scale(self, x):
        x = x + 1
        x = x * ((self.n_categories + 1) / 2.)
        return x

    def _one_hot(self, x):
        y = Variable(torch.arange(0, self.n_categories + 1))
        x = -(((x[..., None] - y) ** 2) * 1e12)
        x = F.softmax(x, dim=-1)
        return x

    def categorical(self, x):
        x = x.view(-1)
        x = self._mu_law(x)
        x = self._shift_and_scale(x)
        x = self._one_hot(x)
        return x

    def forward(self, x):
        sample_size = x.shape[-1]

        # one-hot encode the continuous samples
        x = self.categorical(x)

        # embed the categorical variables into a
        # dense vector
        x = self.linear(x)

        # place all embeddings on the unit sphere
        norms = torch.norm(x, dim=-1)
        x = x / norms.view(-1, 1)
        x = x.view(-1, self.embedding_dim, sample_size)
        return x
