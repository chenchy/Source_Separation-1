import torch
import random
import librosa
import numpy as np

class Compose(object):
    """Composes several augmentation transforms.
    Args:
        augmentations: list of augmentations to compose.
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, audio):
        for t in self.transforms:
            audio = t(audio)

        return audio


def _augment_gain(audio, low=0.25, high=1.25):
    """Applies a random gain between `low` and `high`"""
    g = low + torch.rand(1) * (high - low)
    return audio * g


def _augment_channelswap(audio):
    """Swap channels of stereo signals with a probability of p=0.5"""
    if audio.shape[0] == 2 and torch.FloatTensor(1).uniform_() < 0.5:
        return torch.flip(audio, [0])
    else:
        return audio

def _augment_freq_masking(spec, n_freq_mask=100):
    n_features = spec.shape[2]
    # mask freq
    f = np.random.uniform(low=0.0, high=n_freq_mask)
    f = int(f)
    f0 = random.randint(0, n_features-f)
    spec[:, :, f0:f0+f, :] = 0
    return spec

def _augment_time_warp(spec, W=5):
    from specAugment.sparse_image_warp_pytorch import sparse_image_warp

    num_rows = spec.shape[2]
    spec_len = spec.shape[3]

    y = num_rows // 2
    horizontal_line_at_ctr = spec[0][y]

    point_to_warp = horizontal_line_at_ctr[random.randrange(W, spec_len-W)]

    # Uniform distribution from (0,W) with chance to be up to W negative
    dist_to_warp = random.randrange(-W, W)
    src_pts = torch.tensor([[[y, point_to_warp]]])
    dest_pts = torch.tensor([[[y, point_to_warp + dist_to_warp]]])
    warped_spectro, dense_flows = sparse_image_warp(spec, src_pts, dest_pts)
    return warped_spectro.squeeze(3)

def _augment_pitchShift(audio):
    """Randomly shift pitch"""
    audio = audio.cpu().detach().numpy()
    step = random.choice([-4, -3, -2, -1, 0, 0, 0, 0, 1, 2, 3, 4])
    audio[0] = librosa.effects.pitch_shift(audio[0], 44100, n_steps=step)
    audio[1] = librosa.effects.pitch_shift(audio[1], 44100, n_steps=step)
    return torch.from_numpy(audio).float()