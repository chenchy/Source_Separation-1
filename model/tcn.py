import torch.nn as nn
import torch.nn.functional as F

class ConvBlock(nn.Module):
    def __init__(self, in_channels, oup_channels, kernel_size, dilate):
        super().__init__()

        pad = dilate * (kernel_size -1) // 2

        self.conv1 = nn.Conv1d(in_channels, oup_channels, kernel_size, padding=pad, dilation=dilate, groups=oup_channels)
        self.norm1 = nn.BatchNorm1d(oup_channels)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(oup_channels, oup_channels, kernel_size, padding=pad, dilation=dilate, groups=oup_channels)
        self.norm2 = nn.BatchNorm1d(oup_channels)
        self.relu2 = nn.ReLU()

        self.conv_res = nn.Conv1d(in_channels, oup_channels, 1)
        self.act = nn.ReLU()

    def forward(self, inp):
        '''
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_mel, nb_timesteps)
        '''
        hid = self.relu1(self.norm1(self.conv1(inp)))
        hid = self.relu2(self.norm2(self.conv2(hid)))
        res = self.conv_res(inp)
        oup = res + hid
        oup = self.act(oup)

        return oup, hid

class tcn(nn.Module):
    def __init__(self, inp_features, n_hidden, output_features, kernal_size, n_stacks, n_blocks, max_bin):
        super().__init__()

        self.n_features = inp_features

        self.input_norm = nn.BatchNorm1d(self.n_features)
        self.conv_in = nn.Conv1d(self.n_features, n_hidden, 1)

        di_conv_layers = []
        for i_repeat in range(n_stacks):
            for i_block in range(n_blocks):
                di_conv_layers.append(
                    ConvBlock(n_hidden, n_hidden, kernal_size, 2**i_block))
        self.di_conv_layers = nn.ModuleList(di_conv_layers)

        self.prelu_out = nn.PReLU()
        self.max_bin = max_bin

        self.linear = nn.Linear(n_hidden, output_features)

    def forward(self, inp, mix):
        '''
        Input: (nb_samples, nb_channels, nb_features, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_features, nb_timesteps)
        '''
        inp = inp[:, :, :self.max_bin]
        nb_samples, nb_channels, nb_features, nb_timesteps = inp.shape
        inp = inp.reshape(-1, nb_features, nb_timesteps)#[:, :-1]
        output = self.conv_in(self.input_norm(inp))
        skip_sum = 0.0
        for di_conv_layer in self.di_conv_layers:
            residual, skip = di_conv_layer(output)
            output = output + residual
            skip_sum = skip_sum + skip
        output = self.prelu_out(skip_sum)
        output = self.linear(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = output.reshape(nb_samples, nb_channels, -1, nb_timesteps)

        oup = F.sigmoid(output) * mix

        return oup
