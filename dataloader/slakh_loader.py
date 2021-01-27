import torch
import musdb
import random
import pretty_midi
import pypianoroll
from utils.general_utils import target_to_midi_number
import os
import warnings
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
            self.file_list = np.load('valid_bass.npy')[49:]
        else:
            self.file_list = np.load('valid_bass.npy')[:49]

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

            if len(midi_data.instruments) == 0 or all(len(i.notes) == 0 for i in midi_data.instruments):
                index = index - 1 if index > 0 else index + 1
                return self.__getitem__(index)

            # sythesized
            sf2_name = random.choice(self.sf2_list)
            audio_mix, norm_max = self.fluidsynth(midi_data, self.sr, self.sf2_dir + sf2_name)
            audio_tar, _ = self.fluidsynth(target_data, self.sr, self.sf2_dir + sf2_name, norm_max)
            if len(audio_tar) < self.sr * self.samples_per_track * self.seq_duration or len(audio_mix) < self.sr * self.samples_per_track * self.seq_duration:
                index = index - 1 if index > 0 else index + 1
                return self.__getitem__(index)

            audio_mix_list, audio_tar_list = [], []
            chunk = self.seq_duration * self.sr
            for i in range(self.samples_per_track):
                audio_mix_list.append(audio_mix[chunk * i : chunk * (i+1)])
                audio_tar_list.append(audio_tar[chunk * i : chunk * (i+1)])
            audio_mix = np.array(audio_mix_list)
            audio_tar = np.array(audio_tar_list)
            
        if self.split == 'valid':
            audio_tar = sf.read('../data/slakh/val_audio/tar/'+file_name.replace('mid', 'wav'), stop=180*44100)[0][None, :] #, audio_tar.T, 44100)
            audio_mix = sf.read('../data/slakh/val_audio/mix/'+file_name.replace('mid', 'wav'), stop=180*44100)[0][None, :] #, audio_mix.T, 44100)
        
        return audio_mix, audio_tar, file_name

    def fluidsynth(self, pretty_midi_obj, fs, sf2_path, norm_max=None):
        #if len(pretty_midi_obj.instruments) == 0 or all(len(i.notes) == 0 for i in pretty_midi_obj.instruments):
        #    print('yes!')
        # Get synthesized waveform for each instrument
        waveforms = [i.fluidsynth(fs=fs, sf2_path=sf2_path) for i in pretty_midi_obj.instruments]
        # Allocate output waveform, with #sample = max length of all waveforms
        synthesized = np.zeros(np.max([w.shape[0] for w in waveforms]))
        # Sum all waveforms in
        for waveform in waveforms:
            synthesized[:waveform.shape[0]] += waveform
        # Normalize
        if not norm_max:    
            norm_max = np.abs(synthesized).max()
        #synthesized = float(synthesized)
        synthesized /= norm_max

        #print(synthesized.max(), synthesized.min())
        return synthesized, norm_max

    def __len__(self):
        return len(self.file_list) #// 2 #* self.samples_per_track