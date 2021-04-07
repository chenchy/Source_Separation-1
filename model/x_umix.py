from torch.nn import LSTM, Linear, BatchNorm1d, Parameter, Tanh, ReLU
import torch
import torch.nn as nn
import torch.nn.functional as F

class STFT(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        center=False
    ):
        super(STFT, self).__init__()
        self.window = nn.Parameter(
            torch.hann_window(n_fft),
            requires_grad=False
        )
        self.n_fft = n_fft
        self.n_hop = n_hop
        self.center = center

    def forward(self, x):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_bins, nb_frames, 2)
        """

        nb_samples, nb_channels, nb_timesteps = x.size()

        # merge nb_samples and nb_channels for multichannel stft
        x = x.reshape(nb_samples*nb_channels, -1)

        # compute stft with parameters as close as possible scipy settings
        stft_f = torch.stft(
            x,
            n_fft=self.n_fft, hop_length=self.n_hop,
            window=self.window, center=self.center,
            normalized=False, onesided=True,
            pad_mode='reflect'
        )

        # reshape back to channel dimension
        stft_f = stft_f.contiguous().view(
            nb_samples, nb_channels, self.n_fft // 2 + 1, -1, 2
        )
        return stft_f


class Spectrogram(nn.Module):
    def __init__(
        self,
        power=1,
        mono=True
    ):
        super(Spectrogram, self).__init__()
        self.power = power
        self.mono = mono

    def forward(self, stft_f):
        """
        Input: complex STFT
            (nb_samples, nb_bins, nb_frames, 2)
        Output: Power/Mag Spectrogram
            (nb_frames, nb_samples, nb_channels, nb_bins)
        """
        stft_f = stft_f.transpose(2, 3)
        # take the magnitude
        stft_f = stft_f.pow(2).sum(-1).pow(self.power / 2.0)

        # downmix in the mag domain
        if self.mono:
            stft_f = torch.mean(stft_f, 1, keepdim=True)

        # permute output for LSTM convenience
        return stft_f.permute(2, 0, 1, 3)

class Layer(nn.Module):
    def __init__(self, inp, oup, act='relu'):
        super(Layer1, self).__init__()
        if act == 'relu':
            self.act = ReLU()
        elif act == 'tanh':
            self.act = Tanh()
        self.inp_features = inp
        self.oup_features = oup
        self.layer = nn.Sequential(
            Linear(inp, oup, bias=False),
            BatchNorm1d(oup),
            self.act
        )

    def forward(self, x):
        return self.layer(x)

class lstm(nn.Module):
    def __init__(self, hidden_size, nb_layers, unidirectional):
        super(Layer1, self).__init__()
        self.lstm = LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size, 
            num_layers=nb_layers, 
            bidirectional=not unidirectional, 
            batch_first=False, 
            dropout=0.4
        )

    def forward(self, x):
        return self.lstm(x)

class OpenUnmix(nn.Module):
    def __init__(
        self,
        n_fft=4096,
        n_hop=1024,
        input_is_spectrogram=False,
        hidden_size=512,
        nb_channels=2,
        sample_rate=44100,
        nb_layers=3,
        input_mean=None,
        input_scale=None,
        max_bin=None,
        unidirectional=False,
        power=1,
        n_sources=4
    ):
        """
        Input: (nb_samples, nb_channels, nb_timesteps)
            or (nb_frames, nb_samples, nb_channels, nb_bins)
        Output: Power/Mag Spectrogram
                (nb_frames, nb_samples, nb_channels, nb_bins)
        """

        super(OpenUnmix, self).__init__()

        self.n_sources = n_sources
        self.nb_output_bins = n_fft // 2 + 1
        if max_bin:
            self.nb_bins = max_bin
        else:
            self.nb_bins = self.nb_output_bins

        self.hidden_size = hidden_size
           
        self.model_dict = {}
        for n in n_sources:
            self.model_dict[n]['input_mean'], self.model_dict[n]['input_scale'], self.model_dict[n]['output_mean'], self.model_dict[n]['output_scale'] = self.get_mean_val(input_mean, input_scale)
            self.model_dict[n]['layer1'] = Layer(self.nb_bins*nb_channels, hidden_size, 'tanh')
            self.model_dict[n]['lstm'] = lstm(hidden_size, nb_layers, bidirectional)
            self.model_dict[n]['layer2'] = Layer(hidden_size*2, hidden_size, 'relu')
            self.model_dict[n]['layer3'] = Layer(hidden_size, self.nb_output_bins*nb_channels)

    def get_mean_val(input_mean, input_scale):
        if input_mean is not None:
            input_mean = torch.from_numpy(
                -input_mean[:self.nb_bins]
            ).float()
        else:
            input_mean = torch.zeros(self.nb_bins)

        if input_scale is not None:
            input_scale = torch.from_numpy(
                1.0/input_scale[:self.nb_bins]
            ).float()
        else:
            input_scale = torch.ones(self.nb_bins)

        input_mean = Parameter(input_mean)
        input_scale = Parameter(input_scale)

        output_scale = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        output_mean = Parameter(
            torch.ones(self.nb_output_bins).float()
        )
        return input_mean, input_scale, output_mean, output_scale

    def forward(self, x_mag, mix, x_pha):
       
        x = x.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # crop
        x = x[..., :self.nb_bins]

        input_list = [x.copy(), x.copy(), x.copy(), x.copy()]

        # shift and scale input to mean=0 std=1 (across all bins)
        for i in range(self.n_sources):
            input_list[i] += self.model_dict[i]['input_mean']
            input_list[i] *= self.model_dict[i]['input_scale']
            input_list[i] = self.model_dict[i]['layer1'](input_list[i].reshape(-1, nb_channels*self.nb_bins))
            input_list[i] = input_list[i].reshape(nb_frames, nb_samples, self.hidden_size)

        cross_1 = (input_list[0] + input_list[1] + input_list[2] + input_list[4]) / 4.0

        for i in range(self.n_sources):
            input_list[i] = torch.cat([input_list[i], self.model_dict[i]['lstm'](cross_1)[0]], -1)
            input_list[i] = self.model_dict[i]['layer2'](input_list[i].reshape(-1, input_list[i].shape[-1]))
            input_list[i] = self.model_dict[i]['layer3'](input_list[i]).reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
            input_list[i] *= self.model_dict[i]['output_scale']
            input_list[i] += self.model_dict[i]['output_mean']
            input_list[i] = F.relu(input_list[i]).permute(1, 2, 3, 0) * mix

        return input_list[i]