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
import soundfile as sf

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
        sr=44100,
        tempo=100.0
    ):

        random.seed(seed)

        self.root = root
        self.sf2_dir = sf2_dir
        self.sf2_list = os.listdir(sf2_dir)
        self.seq_duration = seq_duration
        self.target = target
        self.samples_per_track = samples_per_track
        self.split = split
        self.tempo = tempo
        self.sr = sr
        if self.split == 'train':
            self.file_list = np.load('valid_bass.npy')[50:]
        else:
            self.file_list = np.load('valid_bass.npy')[:50]

    def __getitem__(self, index):

        #index = index // self.samples_per_track

        file_name = self.file_list[index]

        if self.split == 'train':
            midi_data = pretty_midi.PrettyMIDI(os.path.join(self.root, file_name))

            # add see instrument / no longer then
            
            midi_data = pypianoroll.from_pretty_midi(midi_data)
            midi_data.tempo = np.full(midi_data.tempo.shape, self.tempo) 

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
        
            
            length = target_data.get_length()
            resolution = int(24 / (60 / self.tempo)) * self.samples_per_track * self.seq_duration
            chunk_start = random.uniform(0, length - resolution)
            midi_data = midi_data.trim(0, int(chunk_start) + resolution)
            midi_data = midi_data.trim(int(chunk_start), int(chunk_start) + resolution)
            midi_data = pypianoroll.to_pretty_midi(midi_data)
            target_data = target_data.trim(0, int(chunk_start) + resolution)
            target_data = target_data.trim(int(chunk_start), int(chunk_start) + resolution)
            target_data = pypianoroll.to_pretty_midi(target_data)

            # sythesized
            sf2_name = random.choice(self.sf2_list)
            audio_mix = midi_data.fluidsynth(self.sr, self.sf2_dir + sf2_name)
            audio_tar = target_data.fluidsynth(self.sr, self.sf2_dir + sf2_name)
            if len(audio_tar) < self.sr * self.samples_per_track * self.seq_duration or len(audio_mix) < self.sr * self.samples_per_track * self.seq_duration:
                index = index - 1 if index > 0 else index + 1
                return self.__getitem__(index)
            
            audio_mix_list, audio_tar_list = [], []
            chunk = self.seq_duration * self.sr
            for i in range(self.samples_per_track):
                audio_mix_list.append(audio_mix[chunk * i : chunk * (i+1)])
                audio_tar_list.append(audio_tar[chunk * i : chunk * (i+1)])
                #chunk_start = int(random.uniform(0, length - self.seq_duration * self.sr))
                #audio_mix_list.append(audio_mix[chunk_start : chunk_start + self.seq_duration * self.sr])
                #audio_tar_list.append(audio_tar[chunk_start : chunk_start + self.seq_duration * self.sr])
            audio_mix = np.array(audio_mix_list)
            audio_tar = np.array(audio_tar_list)
            
        if self.split == 'valid':
            #length = min(audio_mix.shape[-1], audio_tar.shape[-1])
            #audio_tar = audio_tar[None, :length]
            #audio_mix = audio_mix[None, :length]
            audio_tar = sf.read('../data/slakh/val_audio/tar/'+file_name.replace('mid', 'wav'), stop=180*44100)[0][None, :] #, audio_tar.T, 44100)
            audio_mix = sf.read('../data/slakh/val_audio/mix/'+file_name.replace('mid', 'wav'), stop=180*44100)[0][None, :] #, audio_mix.T, 44100)

        '''
        # chunk data
        
        if self.split == 'train':
            length = target_data.get_length()
            
            chunk_start = random.uniform(0, length - self.seq_duration * 40)
            midi_data = midi_data.trim(0, int(chunk_start) + self.seq_duration * 40)
            midi_data = midi_data.trim(int(chunk_start), int(chunk_start) + self.seq_duration * 40)
            midi_data = pypianoroll.to_pretty_midi(midi_data)
            #print(target_data.get_length(), chunk_start)
            target_data = target_data.trim(0, int(chunk_start) + self.seq_duration * 40)
            target_data = target_data.trim(int(chunk_start), int(chunk_start) + self.seq_duration * 40)
            target_data = pypianoroll.to_pretty_midi(target_data)
        
        sf2_name = random.choice(self.sf2_list)
        audio_mix = midi_data.fluidsynth(self.sr, self.sf2_dir + sf2_name)
        audio_tar = target_data.fluidsynth(self.sr, self.sf2_dir + sf2_name)
        if len(audio_tar) == 0:
            index = index - 1 if index > 0 else index + 1
            return self.__getitem__(index)
        '''
        return audio_mix, audio_tar, file_name

    def __len__(self):
        return len(self.file_list) #* self.samples_per_track