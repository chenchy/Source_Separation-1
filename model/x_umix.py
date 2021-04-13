from torch.nn import LSTM, Linear, BatchNorm1d, Parameter, Tanh, ReLU, Identity
import torch
import torch.nn as nn
import torch.nn.functional as F

class NoOp(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return x

class Layer(nn.Module):
    def __init__(self, inp, oup, act=None):
        super(Layer, self).__init__()
        if act == 'relu':
            act = ReLU()
        elif act == 'tanh':
            act = Tanh()
        else:
            act = NoOp()
        self.inp_features = inp
        self.oup_features = oup
        self.layer = nn.Sequential(
            Linear(inp, oup, bias=False),
            BatchNorm1d(oup),
            act
        )

    def forward(self, x):
        return self.layer(x)

class lstm(nn.Module):
    def __init__(self, hidden_size, nb_layers, unidirectional):
        super(lstm, self).__init__()
        self.lstm = LSTM(
            input_size=hidden_size, 
            hidden_size=hidden_size//2, 
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
        n_sources=4,
        device='cuda'
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
           
        
        self.model_dict = nn.ModuleDict()
        self.param_dict = {}
        for n in range(n_sources):
            n = str(n)
            self.model_dict[n] = nn.ModuleDict()
            self.param_dict[n] = {}
            self.param_dict[n]['input_mean'], self.param_dict[n]['input_scale'], self.param_dict[n]['output_mean'], self.param_dict[n]['output_scale'] = self.get_mean_val(input_mean, input_scale, device)
            self.model_dict[n]['id'] = Identity()
            self.model_dict[n]['layer1'] = Layer(self.nb_bins*nb_channels, hidden_size, 'tanh')
            self.model_dict[n]['lstm'] = lstm(hidden_size, nb_layers, unidirectional)
            self.model_dict[n]['layer2'] = Layer(hidden_size*2, hidden_size, 'relu')
            self.model_dict[n]['layer3'] = Layer(hidden_size, self.nb_output_bins*nb_channels)
        #self.layer2 = Layer(hidden_size*2, hidden_size, 'relu')
        #self.layer3 = Layer(hidden_size, self.nb_output_bins*nb_channels)
        #self.emb = nn.Linear(128, 1024)

    def get_mean_val(self, input_mean, input_scale, device='cuda'):
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
        return input_mean.to(device), input_scale.to(device), output_mean.to(device), output_scale.to(device)

    def forward(self, x_mag, mix, vgg=None):
       
        x = x_mag.permute(3, 0, 1, 2)
        nb_frames, nb_samples, nb_channels, nb_bins = x.data.shape

        # crop
        x = x[..., :self.nb_bins]

        input_list = []
        for i in range(self.n_sources):
            input_list.append(self.model_dict[str(i)]['id'](x))

        # shift and scale input to mean=0 std=1 (across all bins)
        for i in range(self.n_sources):
            input_list[i] += self.param_dict[str(i)]['input_mean']
            input_list[i] *= self.param_dict[str(i)]['input_scale']
            input_list[i] = self.model_dict[str(i)]['layer1'](input_list[i].reshape(-1, nb_channels*self.nb_bins))
            input_list[i] = input_list[i].reshape(nb_frames, nb_samples, self.hidden_size)

        cross_1 = (input_list[0] + input_list[1] + input_list[2] + input_list[3]) / 4.0

        lstm_oup = []
        for i in range(self.n_sources):
            lstm_oup.append(self.model_dict[str(i)]['lstm'](cross_1)[0])
        

        cross_2 = (lstm_oup[0] + lstm_oup[1] + lstm_oup[2] + lstm_oup[3]) / 4.0
         
        emb_oup = []
        for i in range(self.n_sources):
            emb_oup.append(torch.cat([input_list[i], cross_2], -1))

        for i in range(self.n_sources):
            #vgg_inp = self.emb(vgg[:, i]).repeat(1, 43, 1).permute(1, 0, 2)[1:-2]
            input_list[i] = self.model_dict[str(i)]['layer2'](emb_oup[i].reshape(-1, emb_oup[i].shape[-1]))
            input_list[i] = self.model_dict[str(i)]['layer3'](input_list[i]).reshape(nb_frames, nb_samples, nb_channels, self.nb_output_bins)
            input_list[i] *= self.param_dict[str(i)]['output_scale']
            input_list[i] += self.param_dict[str(i)]['output_mean']
            input_list[i] = (F.relu(input_list[i]).permute(1, 2, 3, 0) * mix).unsqueeze(1)
        emb = emb_oup.copy()
        return torch.cat(input_list, 1), emb
      
