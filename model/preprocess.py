import torch.nn as nn
import torch

class Vggish(nn.Module):
    def __init__(self):
        super(Vggish, self).__init__()

        self.vgg = torch.hub.load('harritaylor/torchvggish', 'vggish')
        self.vgg.preprocess = False
        self.vgg.eval()

    def forward(self, inp):
        feature = self.vgg(inp)
        return feature

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

class vgg_mel(nn.Module):
    def __init__(self):
        super(vgg_mel, self).__init__()
        import torchaudio
        import numpy as np

        fft_length = 2 ** int(np.ceil(np.log(400) / np.log(2.0)))
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000,
            n_fft=fft_length,
            win_length=400,
            hop_length=160,
            n_mels=64,
            f_min=125,
            f_max=7500,
            window_fn=torch.hann_window,
            wkwargs={'periodic':True}
        )
        self.resample = torchaudio.transforms.Resample(44100, 16000)

    def forward(self, data, sample_rate):
        '''
        Input: (nb_samples, nb_channels, nb_timesteps)
        Output:(nb_samples, nb_channels, nb_mel, nb_timesteps)
        '''
    
        mel = self.mel(data)

        log_mel = torch.log(mel + 0.01)

        return log_mel

if __name__ == '__main__':
    import os
    import tqdm
    import soundfile as sf
    import librosa
    import numpy as np

    _dir = '../../data/MUSDB18-HQ/train/'
    vgg = Vggish()
    vgg_mag = vgg_mel()
    for file in tqdm.tqdm(os.listdir(_dir)):
        if not os.path.exists('../../data/MUSDB18-HQ/vggish/'+file+'.npy'):
            wav_data, sr = sf.read(os.path.join(_dir, file, 'vocals.wav'))
            if sr != 16000:
                wav_data = librosa.resample(wav_data.T, sr, 16000)
            x = vgg_mag(torch.from_numpy(wav_data).unsqueeze(0).float(), sr).squeeze(0)
            trim = int((x.shape[-1]//96)*96)
            x = x[..., :trim]
            x = x.permute(0, 2, 1).reshape(2, -1, 96, 64).reshape(-1, 96, 64)[:, None, :, :]
            feature = vgg(x)
            feature = feature.reshape(2, -1, feature.shape[-1]).permute(0, 2, 1)
            np.save('../../data/MUSDB18-HQ/vggish/'+file, feature.detach().cpu().numpy())

