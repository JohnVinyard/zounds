import numpy as np
import featureflow as ff
from torch import nn
from torch.nn import functional as F


def sigmoid(a):
    return 1. / (1 + np.exp(-a))


def stochastic_binary(a):
    return a > np.random.random_sample(a.shape)


def simple_settings(cls):
    """
    Create sane default persistence settings for learning pipelines
    :param cls: The class to decorate
    """

    class Settings(ff.PersistenceSettings):
        _id = cls.__name__
        id_provider = ff.StaticIdProvider(_id)
        key_builder = ff.StringDelimitedKeyBuilder()
        database = ff.FileSystemDatabase(
            path=_id, key_builder=key_builder, createdirs=True)

    class Model(cls, Settings):
        pass

    return Model


class ConvLayer(nn.Module):
    def __init__(
            self,
            layer_type,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=lambda x: F.leaky_relu(x, 0.2),
            dropout=True):
        super(ConvLayer, self).__init__()
        self.dropout = dropout
        self.l1 = layer_type(
            in_channels, out_channels, kernel_size, stride, padding, bias=False)

        if '1d' in layer_type.__class__.__name__:
            self.bn = nn.BatchNorm1d(out_channels)
        else:
            self.bn = nn.BatchNorm2d(out_channels)

        self.activation = activation

    def forward(self, x):
        x = self.l1(x)
        x = self.bn(x)
        x = self.activation(x)
        if self.dropout:
            x = F.dropout(x, 0.2)
        return x


class Conv1d(ConvLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout=True):
        super(Conv1d, self).__init__(
            nn.Conv1d,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=lambda x: F.leaky_relu(x, 0.2),
            dropout=dropout)


class ConvTranspose1d(ConvLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=lambda x: F.leaky_relu(x, 0.2),
            dropout=True):
        super(ConvTranspose1d, self).__init__(
            nn.ConvTranspose1d,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=activation,
            dropout=dropout)


class Conv2d(ConvLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout=True):
        super(Conv2d, self).__init__(
            nn.Conv2d,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=lambda x: F.leaky_relu(x, 0.2),
            dropout=dropout)


class ConvTranspose2d(ConvLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=lambda x: F.leaky_relu(x, 0.2),
            dropout=True):
        super(ConvTranspose2d, self).__init__(
            nn.ConvTranspose2d,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=activation,
            dropout=dropout)
