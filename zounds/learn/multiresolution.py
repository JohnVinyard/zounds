from torch import nn
from gated import GatedConvLayer, GatedConvTransposeLayer


class MultiResolutionBlock(nn.Module):
    """
    A layer that convolves several different filter/kernel sizes with the same
    input features
    """

    def __init__(
            self,
            layer,
            in_channels,
            out_channels,
            kernel_sizes,
            stride=1,
            padding=None,
            **kwargs):
        super(MultiResolutionBlock, self).__init__()

        layers = [
            layer(
                in_channels,
                out_channels,
                k,
                stride,
                padding=k // 2 if padding is None else padding,
                **kwargs)
            for k in kernel_sizes]

        self.main = nn.Sequential(*layers)
        self.in_channels = in_channels
        self.out_channels = out_channels

    def forward(self, x):
        results = []
        for m in self.main:
            r = m(x)
            results.append(r)
        return results


class MultiResolutionConvLayer(MultiResolutionBlock):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_sizes,
            stride=1,
            padding=None,
            **kwargs):

        super(MultiResolutionConvLayer, self).__init__(
            GatedConvLayer,
            in_channels,
            out_channels,
            kernel_sizes,
            stride,
            padding,
            **kwargs)


class MultiResolutionConvTransposeLayer(MultiResolutionBlock):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_sizes,
            stride=1,
            padding=None):

        super(MultiResolutionConvTransposeLayer, self).__init__(
            GatedConvTransposeLayer,
            in_channels,
            out_channels,
            kernel_sizes,
            stride,
            padding)
