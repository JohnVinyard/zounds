from __future__ import print_function, division
import numpy as np
import featureflow as ff
import torch
from torch import nn
from torch.nn import functional as F
from torch.autograd import Variable
import hashlib


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

    Model.__name__ = cls.__name__
    Model.__module__ = cls.__module__
    return Model


def object_store_pipeline_settings(container, region, username, api_key):
    def decorator(cls):
        class Settings(ff.PersistenceSettings):
            _id = cls.__name__
            id_provider = ff.StaticIdProvider(_id)
            key_builder = ff.StringDelimitedKeyBuilder()
            database = ff.ObjectStoreDatabase(
                container, username, api_key, region, key_builder=key_builder)

        class Model(cls, Settings):
            pass

        Model.__name__ = cls.__name__
        Model.__module__ = cls.__module__
        return Model

    return decorator


def trainable_parameters(model):
    return filter(lambda x: x.requires_grad, model.parameters())


def model_hash(model):
    h = hashlib.md5()
    h.update(str(model))
    for p in model.parameters():
        h.update(p.data.cpu().numpy())
    return h.hexdigest()


def gradients(network):
    for n, p in network.named_parameters():
        g = p.grad
        if g is None:
            continue
        yield n, g.min().data[0], g.max().data[0], g.mean().data[0]


def to_var(x, volatile=False):
    t = torch.from_numpy(x)
    v = Variable(t, volatile=volatile).cuda()
    return v


def from_var(x):
    return x.data.cpu().numpy()


def try_network(network, x, **kwargs):
    network_is_cuda = next(network.parameters()).is_cuda

    x = Variable(torch.from_numpy(x), volatile=True)
    if network_is_cuda:
        x = x.cuda()

    result = network(x, **kwargs)
    return result


def apply_network(network, x, chunksize=None):
    """
    Apply a pytorch network, potentially in chunks
    """
    network_is_cuda = next(network.parameters()).is_cuda

    x = torch.from_numpy(x)

    with torch.no_grad():
        if network_is_cuda:
            x = x.cuda()

        if chunksize is None:
            return from_var(network(x))

        return np.concatenate(
            [from_var(network(x[i: i + chunksize]))
             for i in xrange(0, len(x), chunksize)])


def sample_norm(x):
    """
    pixel norm as described in section 4.2 here:
    https://arxiv.org/pdf/1710.10196.pdf
    """
    original = x
    # square
    x = x ** 2
    # feature-map-wise sum
    x = torch.sum(x, dim=1)
    # scale by number of feature maps
    x *= 1.0 / original.shape[1]
    x += 10e-8
    x = torch.sqrt(x)
    return original / x.view(-1, 1, x.shape[-1])


def feature_map_size(inp, kernel, stride=1, padding=0):
    return ((inp - kernel + (2 * padding)) / stride) + 1


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
            dropout=True,
            batch_norm=True,
            dilation=1,
            sample_norm=False):

        super(ConvLayer, self).__init__()
        self.sample_norm = sample_norm
        self.dropout = dropout
        self.l1 = layer_type(
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            bias=False,
            dilation=dilation)

        self.bn = None

        if batch_norm:
            if '1d' in layer_type.__name__:
                self.bn = nn.BatchNorm1d(out_channels)
            else:
                self.bn = nn.BatchNorm2d(out_channels)

        self.activation = activation

    @property
    def out_channels(self):
        return self.l1.out_channels

    @property
    def in_channels(self):
        return self.l1.in_channels

    @property
    def kernel_size(self):
        return self.l1.kernel_size

    @property
    def stride(self):
        return self.l1.stride

    @property
    def padding(self):
        return self.l1.padding

    def forward(self, x):
        x = self.l1(x)

        if self.sample_norm:
            x = sample_norm(x)
        elif self.bn:
            x = self.bn(x)

        if self.activation:
            x = self.activation(x)

        if self.dropout:
            x = F.dropout(x, 0.2, self.training)
        return x


class Conv1d(ConvLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout=True,
            batch_norm=True,
            dilation=1,
            sample_norm=False,
            activation=lambda x: F.leaky_relu(x, 0.2)):
        super(Conv1d, self).__init__(
            nn.Conv1d,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            dilation=dilation,
            sample_norm=sample_norm)


class ConvTranspose1d(ConvLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=lambda x: F.leaky_relu(x, 0.2),
            dropout=True,
            batch_norm=True,
            dilation=1,
            sample_norm=False):
        super(ConvTranspose1d, self).__init__(
            nn.ConvTranspose1d,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            dilation=dilation,
            sample_norm=sample_norm)


class Conv2d(ConvLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            dropout=True,
            batch_norm=True,
            dilation=1,
            sample_norm=False,
            activation=lambda x: F.leaky_relu(x, 0.2), ):
        super(Conv2d, self).__init__(
            nn.Conv2d,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            dilation=dilation,
            sample_norm=sample_norm)


class ConvTranspose2d(ConvLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=lambda x: F.leaky_relu(x, 0.2),
            dropout=True,
            batch_norm=True,
            dilation=1,
            sample_norm=False):
        super(ConvTranspose2d, self).__init__(
            nn.ConvTranspose2d,
            in_channels,
            out_channels,
            kernel_size,
            stride,
            padding,
            activation=activation,
            dropout=dropout,
            batch_norm=batch_norm,
            dilation=dilation,
            sample_norm=sample_norm)
