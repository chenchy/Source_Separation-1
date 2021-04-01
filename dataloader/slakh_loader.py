import torch
import musdb
import random
import pretty_midi
import pypianoroll
from utils.general_utils import target_to_midi_number
import os
import warnings
import fluidsynth
import numpy as np

class SlakhDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        target='vocals',
        root=None,
        sf2_dir = None,
        seq_duration=6,
        samples_per_track=64,
        source_augmentations=lambda audio: audio,
        seed=42,
        split='train',
        sr=44100
    ):

        random.seed(seed)

        self.root = root
        self.sf2_dir = sf2_dir
        self.file_list = os.listdir(root)
        self.sf2_list = os.listdir(sf2_dir)
        self.seq_duration = seq_duration
        self.target = target
        self.samples_per_track = samples_per_track
        self.split = split
        self.sr = sr
       

    def __getitem__(self, index):

        index = index // self.samples_per_track

        file_name = self.file_list[index]

        midi_data = pretty_midi.PrettyMIDI(os.path.join(self.root, file_name))

        # add see instrument / no longer then
        
        midi_data = pypianoroll.from_pretty_midi(midi_data)
        midi_data.tempo = np.full(midi_data.tempo.shape, 100.0) 

        # test target inside
        target_tracks = []
        for inst in midi_data.tracks:
            if inst.program in target_to_midi_number(self.target):
                target_tracks.append(inst.copy().standardize())
        if len(target_tracks) == 0:
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        else: 
            target_data = midi_data.copy()
            target_data.tracks = target_tracks

        # chunk data
        if self.split == 'train':
            length = target_data.get_length()
            if length < 60:
                index = index - 1 if index > 0 else index + 1
                return self.__getitem__(index)
            else:
                chunk_start = random.uniform(0, length - self.seq_duration * 40)
                midi_data = midi_data.trim(0, int(chunk_start) + self.seq_duration * 40)
                midi_data = midi_data.trim(int(chunk_start), int(chunk_start) + self.seq_duration * 40)
                midi_data = pypianoroll.to_pretty_midi(midi_data)
                #print(target_data.get_length(), chunk_start)
                target_data = target_data.trim(0, int(chunk_start) + self.seq_duration * 40)
                target_data = target_data.trim(int(chunk_start), int(chunk_start) + self.seq_duration * 40)
                target_data = pypianoroll.to_pretty_midi(target_data)

        print(target_data.get_end_time(), midi_data.get_end_time())
        # sythesized
        sf2_name = random.choice(self.sf2_list)
        audio_mix = midi_data.fluidsynth(self.sr, self.sf2_dir + sf2_name)
        audio_tar = target_data.fluidsynth(self.sr, self.sf2_dir + sf2_name)
        if len(audio_tar) == 0:
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        print(audio_mix.shape, audio_tar.shape)
        return audio_mix, audio_tar

    def __len__(self):
        return len(self.file_list) * self.samples_per_track