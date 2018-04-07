from torch import nn
from torch.nn import functional as F


class GatedLinearLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(GatedLinearLayer, self).__init__()
        self.l1 = nn.Linear(in_channels, out_channels)
        self.gate = nn.Linear(in_channels, out_channels)

    def forward(self, x):
        return F.tanh(self.l1(x)) * F.sigmoid(self.gate(x))


class GatedLayer(nn.Module):
    def __init__(
            self,
            layer_type,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=0,
            dilation=1,
            attention_func=F.sigmoid,
            activation_func=F.tanh,
            norm=lambda x: x):
        super(GatedLayer, self).__init__()
        self.activation_func = activation_func
        self.norm = norm
        self.conv = layer_type(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.gate = layer_type(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel,
            stride=stride,
            padding=padding,
            dilation=dilation,
            bias=False)
        self.attention_func = attention_func

    def forward(self, x):
        c = self.conv(x)
        c = self.norm(c)
        g = self.gate(x)
        g = self.norm(g)
        out = self.activation_func(c) * self.attention_func(g)
        return out


class GatedConvLayer(GatedLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=0,
            dilation=1,
            attention_func=F.sigmoid,
            activation_func=F.tanh,
            norm=lambda x: x):
        super(GatedConvLayer, self).__init__(
            nn.Conv1d,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            attention_func,
            activation_func,
            norm)


class GatedConvTransposeLayer(GatedLayer):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel,
            stride=1,
            padding=0,
            dilation=1,
            attention_func=F.sigmoid,
            activation_func=F.tanh,
            norm=lambda x: x):
        super(GatedConvTransposeLayer, self).__init__(
            nn.ConvTranspose1d,
            in_channels,
            out_channels,
            kernel,
            stride,
            padding,
            dilation,
            attention_func,
            activation_func,
            norm)
