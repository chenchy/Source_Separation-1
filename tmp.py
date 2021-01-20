import os
import subprocess as sp
import tqdm
import pretty_midi
import pypianoroll
import tqdm
import numpy as np
from utils.general_utils import target_to_midi_number

def select_inst(inst_name):
    valid_midi = []
    for file in tqdm.tqdm(os.listdir('../data/slakh/midi/')):
        try:
            midi_data = pretty_midi.PrettyMIDI(os.path.join('../data/slakh/midi/', file))
            
            midi_data = pypianoroll.from_pretty_midi(midi_data)

            # test target inside
            target_tracks = []
            max_length = 0
            for inst in midi_data.tracks:
                if inst.program in target_to_midi_number(inst_name):
                    max_length = max(max_length, inst.get_length())
                    target_tracks.append(inst.copy().standardize())
            if len(target_tracks) != 0 and max_length > 40 * 96:
                valid_midi.append(file)
                print(len(valid_midi))
        except Exception as e: 
            pass
    print(len(valid_midi), len(os.listdir('../data/slakh/midi/')))
    np.save('valid_bass.npy', np.array(valid_midi))

select_inst('bass')

def valid_synth(file_path):
    for file in tqdm.tqdm(np.load(file_path)[:1]):
        
        try:
            midi_data = pretty_midi.PrettyMIDI(os.path.join('../data/slakh/midi/', file))
            audio = midi_data.fluidsynth(44100, '../data/sf2/FluidR3_GM.sf2')
            print(audio.shape, audio.max(), audio.min())
        except Exception as e: 
            print(file, e)
        '''
        print(file)
        command = [
            'fluidsynth',
            '-a', 'alsa',
            '-m', 'alsa_seq',
            '../data/sf2/FluidR3_GM.sf2',
            '../data/slakh/midi/'+file
        ]
        command.append('-')
        p = sp.Popen(command, stdout=sp.PIPE, stderr=sp.PIPE)
        bytes_per_sample = np.dtype(np.int16).itemsize
        frame_size = bytes_per_sample
        chunk_size = frame_size * 44100
        with p.stdout as stdout:
            print(stdout)
            data = stdout.read(chunk_size)
            print(data)
        '''

#valid_synth('valid_bass.npy')