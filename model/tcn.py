import torch.nn as nn
import torch.nn.functional as F
import torch

class ConvBlock(nn.Module):
    def __init__(self, in_channels, oup_channels, kernel_size, dilate, groups=512):
        super().__init__()

        pad = dilate * (kernel_size - 1) // 2

        self.conv1x1 = nn.Conv1d(in_channels, oup_channels, 1) #, groups=groups)
        self.norm1 = nn.BatchNorm1d(oup_channels)
        self.prelu1 = nn.PReLU()
        self.conv_d = nn.Conv1d(oup_channels, oup_channels, kernel_size, padding=pad, dilation=dilate, groups=oup_channels)
        self.prelu2 = nn.PReLU()
        self.norm2 = nn.BatchNorm1d(oup_channels)
        self.conv_out = nn.Conv1d(oup_channels, in_channels, 1)
        self.conv_skip = nn.Conv1d(oup_channels, in_channels, 1)

    def forward(self, inp):
        '''
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_mel, nb_timesteps)
        '''
        hid = self.conv1x1(inp)
        hid = self.prelu1(hid)
        hid = self.norm1(hid)
        hid = self.conv_d(hid)
        hid = self.prelu2(hid)
        hid = self.norm2(hid)
        out = self.conv_out(hid)
        skip = self.conv_skip(hid)
        return out, skip

class tcn(nn.Module):
    def __init__(self, inp_features, n_hidden, output_features, kernal_size, n_stacks, n_blocks, max_bin, input_mean=None, input_scale=None):
        super().__init__()

        self.n_features = inp_features #- 1

        self.max_bin = max_bin

        self.input_norm = nn.BatchNorm1d(max_bin*2)
        self.conv_in = nn.Conv1d(max_bin*2, n_hidden, 1)

        di_conv_layers = []
        for i_repeat in range(n_stacks):
            for i_block in range(n_blocks):
                di_conv_layers.append(
                    ConvBlock(n_hidden, 128, 3, 2**i_block))
        self.di_conv_layers = nn.ModuleList(di_conv_layers)
        self.prelu_out = nn.PReLU()

        self.linear = nn.Linear(n_hidden, output_features*2)

        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:inp_features]
            ).float()
        else:
            input_mean = torch.zeros(inp_features)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:inp_features]
            ).float()
        else:
            input_scale = torch.ones(inp_features)

        self.input_mean = nn.Parameter(input_mean)
        self.input_scale = nn.Parameter(input_scale)

        self.output_scale = nn.Parameter(
            torch.ones(output_features).float()
        )
        self.output_mean = nn.Parameter(
            torch.ones(output_features).float()
        )

    def forward(self, inp, mix, emb=None):
        '''
        Input: (nb_samples, nb_channels, nb_features, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_features, nb_timesteps)
        '''
        inp = inp[:, :, :self.max_bin]
        nb_samples, nb_channels, nb_features, nb_timesteps = inp.shape

        inp += self.input_mean[None, None, :, None]
        inp *= self.input_scale[None, None, :, None]

        inp = inp.reshape(nb_samples, -1, nb_timesteps)#[:, :-1]
        #inp = inp.reshape(-1, nb_features, nb_timesteps)#[:, :-1]
        output = self.conv_in(self.input_norm(inp))
        skip_sum = 0.0
        for di_conv_layer in self.di_conv_layers:
            residual, skip = di_conv_layer(output)
            output = output + residual
            skip_sum = skip_sum + skip
        output = self.prelu_out(skip_sum)
        output = self.linear(output.permute(0, 2, 1)).permute(0, 2, 1)
        output = output.reshape(nb_samples, nb_channels, -1, nb_timesteps)

        output *= self.output_scale[None, None, :, None]
        output += self.output_mean[None, None, :, None]

        oup = F.relu(output) * mix

        return oup
