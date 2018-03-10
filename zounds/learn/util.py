from __future__ import print_function, division
import numpy as np
import featureflow as ff
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

try:
    import torch
    from torch import nn
    from torch.nn import functional as F
    from torch.autograd import Variable


    def to_var(x, volatile=False):
        t = torch.from_numpy(x)
        v = Variable(t, volatile=volatile).cuda()
        return v


    def from_var(x):
        return x.data.cpu().numpy()


    def try_network(network, x, **kwargs):
        network = network.cuda()
        x = to_var(x)
        result = network(x, **kwargs)
        return result


    def apply_network(network, x, chunksize=None):
        """
        Apply a pytorch network, potentially in chunks
        """
        x = to_var(x, volatile=True)

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


    class FrequencyDecompositionAnalyzer(nn.Module):
        def __init__(
                self,
                first_feature_map_size,
                growth_factor,
                bands,
                bottleneck):

            super(FrequencyDecompositionAnalyzer, self).__init__()
            self.bottleneck = bottleneck
            self.first_feature_map_size = first_feature_map_size
            self.growth_factor = growth_factor
            self.bands = bands
            self.layer_stacks = [[] for _ in bands]
            self.total_bands = sum(bands)

            for i, band in enumerate(bands):
                out_channels = self.first_feature_map_size
                l1 = Conv1d(1, out_channels, 8, 4, 0)
                self.layer_stacks[i].append(l1)
                fms = feature_map_size(
                    band, l1.kernel_size[0], l1.stride[0], l1.padding[0])
                while fms > 1:
                    in_channels = out_channels
                    out_channels = int(out_channels * self.growth_factor)
                    l1 = Conv1d(in_channels, out_channels, 3, 2, 0)
                    self.layer_stacks[i].append(l1)
                    fms = feature_map_size(
                        fms, l1.kernel_size[0], l1.stride[0], l1.padding[0])

            for band, layers in zip(bands, self.layer_stacks):
                for layer in layers:
                    self.add_module(
                        '{band}-{channels}'.format(
                            band=band, channels=layer.out_channels), layer)

            self.total_features = sum(
                [layers[-1].out_channels for layers in self.layer_stacks])
            self.l1 = nn.Linear(
                self.total_features, self.bottleneck, bias=False)
            self.bn1 = nn.BatchNorm1d(self.bottleneck)

        def forward(self, x):
            x = x.view(-1, 1, self.total_bands)

            fms = []
            start_index = 0
            for i, band in enumerate(self.bands):
                stop = start_index + band
                fm = x[..., start_index: stop]
                start_index = stop
                subset = self.layer_stacks[i]
                for s in subset:
                    fm = s(fm)
                fms.append(fm)

            # push the band-wise frequency maps through some linear layers
            flat = torch.cat(fms, dim=1).squeeze()

            x = self.l1(flat)
            x = self.bn1(x)
            x = F.leaky_relu(x, 0.2)
            x = F.dropout(x, 0.2, self.training)

            return x


    class FrequencyDecompositionGenerator(nn.Module):
        def __init__(
                self,
                first_feature_map_size,
                growth_factor,
                bands,
                z_dim):

            super(FrequencyDecompositionGenerator, self).__init__()
            self.first_feature_map_size = first_feature_map_size
            self.growth_factor = growth_factor
            self.bands = bands
            self.z_dim = z_dim
            self.layer_stacks = [[] for _ in bands]

            for i, band in enumerate(bands):
                out_channels = 1
                in_channels = self.first_feature_map_size
                l1 = ConvTranspose1d(
                    in_channels, out_channels, 8, 4, 0, F.tanh, dropout=False)
                self.layer_stacks[i].append(l1)
                fms = feature_map_size(
                    band, l1.kernel_size[0], l1.stride[0], l1.padding[0])
                while fms > 1:
                    out_channels = in_channels
                    in_channels = int(in_channels * self.growth_factor)
                    fms = feature_map_size(
                        fms, 3, 2, 0)
                    in_channels = self.z_dim if fms == 1 else in_channels
                    l1 = ConvTranspose1d(in_channels, out_channels, 3, 2, 0)
                    self.layer_stacks[i].append(l1)

            for band, layers in zip(bands, self.layer_stacks):
                for layer in layers:
                    self.add_module(
                        '{band}-{channels}'.format(
                            band=band, channels=layer.out_channels), layer)

            self.feature_map_sizes = \
                [layers[-1].in_channels for layers in self.layer_stacks]
            self.total_features = sum(self.feature_map_sizes)
            # self.l1 = nn.Linear(self.z_dim, 256, bias=False)
            # self.bn1 = nn.BatchNorm1d(256)
            # self.l2 = nn.Linear(256, self.total_features, bias=False)
            # self.bn2 = nn.BatchNorm1d(self.total_features)

        def forward(self, x):

            # x = x.view(-1, self.z_dim)
            # x = self.l1(x)
            # x = self.bn1(x)
            # x = F.leaky_relu(x, 0.2)
            # x = F.dropout(x, 0.2, self.training)
            #
            # x = self.l2(x)
            # x = self.bn2(x)
            # x = F.leaky_relu(x, 0.2)
            # x = F.dropout(x, 0.2, self.training)
            #
            # x = x.view(-1, self.total_features, 1)

            x = x.view(-1, self.z_dim, 1)

            current = 0
            bands = []
            for i, fms in enumerate(self.feature_map_sizes):
                stop = current + fms
                # fm = x[:, current: stop, :]
                fm = x
                current = stop
                subset = self.layer_stacks[i][::-1]
                for s in subset:
                    fm = s(fm)
                bands.append(fm.view(-1, fm.size()[-1]))

            return torch.cat(bands, dim=1)

except ImportError:
    pass
