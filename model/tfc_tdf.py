import torch
import torch.nn as nn

from abc import ABC


class WI_Module(nn.Module, ABC):
    def __init__(self):
        super().__init__()

    def init_weights(self):
        init_weights_functional(self, self.activation)


def init_weights_functional(module, activation='default'):
    if isinstance(activation, nn.ReLU):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='relu')

    elif activation == 'relu':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='relu')

    elif isinstance(activation, nn.LeakyReLU):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')

    elif activation == 'leaky_relu':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.kaiming_normal_(param, nonlinearity='leaky_relu')

    elif isinstance(activation, nn.Sigmoid):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif activation == 'sigmoid':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif isinstance(activation, nn.Tanh):
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    elif activation == 'tanh':
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)

    else:
        for param in module.parameters():
            if param.dim() > 1:
                nn.init.xavier_normal_(param)


class TFC(WI_Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, in_channels, num_layers, gr, kt, kf, activation):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        activation: activation function
        """
        super(TFC, self).__init__()

        c = in_channels
        self.H = nn.ModuleList()
        for i in range(num_layers):
            self.H.append(
                nn.Sequential(
                    nn.Conv2d(in_channels=c, out_channels=gr, kernel_size=(kf, kt), stride=1,
                              padding=(kt // 2, kf // 2)),
                    nn.BatchNorm2d(gr),
                    activation(),
                )
            )
            c += gr

        self.activation = self.H[-1][-1]

    def forward(self, x):
        """ [B, in_channels, T, F] => [B, gr, T, F] """
        x_ = self.H[0](x)
        for h in self.H[1:]:
            x = torch.cat((x_, x), 1)
            x_ = h(x)

        return x_

class TIF(nn.Module):
    """ [B, in_channels, T, F] => [B, gr, T, F] """

    def __init__(self, channels, f, bn_factor=16, bias=False, min_bn_units=16, activation=nn.ReLU, init_mode=None):

        """
        channels: # channels
        f: num of frequency bins
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers
        activation: activation function
        """

        super(TIF, self).__init__()
        assert init_mode in [None, 'dk']

        self.init_mode = init_mode
        if bn_factor is None:
            self.tif = nn.Sequential(
                nn.Linear(f, f, bias),
                nn.BatchNorm2d(channels, affine=bias),
                activation()
            )

        elif bn_factor == 'None' or bn_factor == 'none':
            self.tif = nn.Sequential(
                nn.Linear(f, f, bias),
                nn.BatchNorm2d(channels, affine=bias),
                activation()
            )

        else:
            bn_units = max(f // bn_factor, min_bn_units)
            self.bn_units = bn_units
            self.tif = nn.Sequential(
                nn.Linear(f, bn_units, bias),
                nn.BatchNorm2d(channels, affine=bias),
                activation(),
                nn.Linear(bn_units, f, bias),
                nn.BatchNorm2d(channels, affine=bias),
                activation()
            )

    def forward(self, x):
        return self.tif(x)

    def init_weights(self):
        if self.init_mode is None:
            init_weights_functional(self, self.tif[-1])
        elif self.init_mode == 'dk':  # domain knowledge
            init_weights_functional(self, self.tif[-1])
            init_with_domain_knowledge(self.tif[0])
            if len(self.tif) > 3:
                init_with_domain_knowledge(self.tif[3])
        else:
            raise NotImplementedError


class TFC_TIF(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor=16, min_bn_units=16, bias=False,
                 activation=nn.ReLU, tic_init_mode=None):
        """
        in_channels: number of input channels
        num_layers: number of densely connected conv layers
        gr: growth rate
        kt: kernel size of the temporal axis.
        kf: kernel size of the freq. axis
        f: num of frequency bins

        below are params for TIF
        bn_factor: bottleneck factor. if None: single layer. else: MLP that maps f => f//bn_factor => f
        bias: bias setting of linear layers

        activation: activation function
        """

        super(TFC_TIF, self).__init__()
        self.tfc = TFC(in_channels, num_layers, f, kt, kf, activation)
        #self.tif = TIF(gr, f, bn_factor, bias, min_bn_units, activation, tic_init_mode)
        #self.activation = self.tif.tif[-1]

    def forward(self, x):
        x = self.tfc(x)

        return x #+ self.tif(x)

    def init_weights(self):
        self.tfc.init_weights()
        self.tif.init_weights()

class TFC_TDF(nn.Module):
    def __init__(self, in_channels, num_layers, gr, kt, kf, f, bn_factor=16, min_bn_units=16, bias=False, activation=nn.ReLU, tic_init_mode=None):
        super(TFC_TDF, self).__init__()
        self.first_conv = nn.Sequential(
            nn.Conv2d(
                in_channels=2,
                out_channels=in_channels,
                kernel_size=(1, 2),
                stride=1
            ),
            nn.BatchNorm2d(in_channels),
            nn.ReLU(),
        )

        self.encoder = TFC_TIF(in_channels, num_layers, gr, kt, kf, f)
        self.decoder = TFC_TIF(in_channels, num_layers, gr, kt, kf, f)

        self.last_conv = nn.Sequential(

            nn.Conv2d(
                in_channels=in_channels,
                out_channels=2,
                kernel_size=(1, 2),
                stride=1,
                padding=(0, 1)
            ),
            nn.ReLU()
        )

    def forward(self, x, mix):
        x = self.first_conv(x)
        x = self.encoder(x)
        x = self.decoder(x)
        x = self.last_conv(x)

        return x * mix
