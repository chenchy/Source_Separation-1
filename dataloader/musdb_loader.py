import torch
import musdb
import random
import os
import numpy as np
from skimage.measure import block_reduce

class MUSDBDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        target='vocals',
        root=None,
        download=False,
        is_wav=False,
        subsets='train',
        split='train',
        seq_duration=6.0,
        samples_per_track=64,
        source_augmentations=lambda audio: audio,
        random_track_mix=False,
        dtype=torch.float32,
        seed=42,
        add_emb=None,
        emb_feature=None,
        *args, **kwargs
    ):

        random.seed(seed)
        self.is_wav = is_wav
        self.seq_duration = seq_duration
        self.target = target
        self.subsets = subsets
        self.split = split
        self.samples_per_track = samples_per_track
        self.source_augmentations = source_augmentations
        self.random_track_mix = random_track_mix
        self.root = root
        self.add_emb = add_emb
        self.emb_feature=emb_feature
        self.mus = musdb.DB(
            root=root,
            is_wav=is_wav,
            split=split,
            subsets=subsets,
            download=download,
            *args, **kwargs
        )
        self.sample_rate = 44100  # musdb is fixed sample rate
        self.dtype = dtype

    def __getitem__(self, index):
        audio_sources = []
        target_ind = None

        # select track
        track = self.mus.tracks[index // self.samples_per_track]

        # at training time we assemble a custom mix
        if self.split == 'train' and self.seq_duration:
            for k, source in enumerate(self.mus.setup['sources']):
                # memorize index of target source
                if source == self.target:
                    target_ind = k

                # select a random track
                #if self.random_track_mix:
                #    track = random.choice(self.mus.tracks)

                # set the excerpt duration
                track.chunk_duration = self.seq_duration
                # set random start position
                track.chunk_start = random.uniform(
                    0, track.duration - self.seq_duration - 1
                )
                # load source audio and apply time domain source_augmentations
                audio = torch.tensor(
                    track.sources[source].audio.T,
                    dtype=self.dtype
                )
                audio = self.source_augmentations(audio)
                audio_sources.append(audio)

            # create stem tensor of shape (source, channel, samples)
            stems = torch.stack(audio_sources, dim=0)
            # # apply linear mix over source index=0
            x = stems.sum(0)
            # get the target stem
            if target_ind is not None:
                y = stems[target_ind]
            # assuming vocal/accompaniment scenario if target!=source
            else:
                vocind = list(self.mus.setup['sources'].keys()).index('vocals')
                # apply time domain subtraction
                y = x - stems[vocind]

        # for validation and test, we deterministically yield the full
        # pre-mixed musdb track
        else:
            # get the non-linear source mix straight from musdb
            x = torch.tensor(
                track.audio.T,
                dtype=self.dtype
            )
            y = torch.tensor(
                track.targets[self.target].audio.T,
                dtype=self.dtype
            )

        if self.add_emb:
            
            if self.emb_feature == 'vggish' or self.emb_feature == 'sidd_att':
                feature_start_time = (int(np.round(track.chunk_start/0.96)))
                feature_end_time = feature_start_time + 6
            elif self.emb_feature == 'salience':
                feature_start_time = (int(np.round(track.chunk_start*self.sample_rate/256))) 
                feature_end_time = feature_start_time + 1020

            if self.split == 'train':
                feature = np.load(os.path.join(self.root, self.emb_feature, track.name+'.npy'))[feature_start_time: feature_end_time].T[None,]
            else:
                feature = np.load(os.path.join(self.root, self.emb_feature, track.name+'.npy')).T[None,]

            if self.emb_feature == 'salience':
                feature = block_reduce(feature, (1, 1, 4), np.mean)
            return x, y, index // self.samples_per_track, feature
        else:
            return x, y, index // self.samples_per_track
        

    def __len__(self):

        return len(self.mus.tracks) * self.samples_per_track