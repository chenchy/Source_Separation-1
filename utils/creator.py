from dataloader.musdb_loader import MUSDBDataset
from dataloader.slakh_loader import SlakhDataset
from utils.augmentation import Compose
from model import tcn, open_unmix, Unet, spleeter
import torch
from utils.augmentation import Compose, _augment_gain, _augment_channelswap, _augment_pitchShift
from model.preprocess import STFT

def preprocess_creator(hparams):

    if hparams.preprocess_name == 'stft':
        preprocess = STFT(hparams.n_fft, hparams.hop_length)

    return preprocess


def model_creator(hparams):
    if hparams.model_name == 'tcn':
        model = tcn.tcn(hparams.max_bin, hparams.n_features, hparams.n_fft//2+1,
                    hparams.kernal_size, hparams.n_stacks, hparams.n_blocks, hparams.max_bin, hparams.mean, hparams.std)

    if hparams.model_name == 'unet':
        model = Unet.Unet(hparams.n_fft, hparams.max_bin, hparams.mean, hparams.std)

    if hparams.model_name == 'spleeter':
        model = spleeter.Spleeter()

    if hparams.model_name == 'open-unmix':
        model = open_unmix.OpenUnmix(n_fft=hparams.n_fft, 
                                    n_hop=hparams.hop_length,
                                    nb_channels=hparams.n_channels,
                                    hidden_size=hparams.n_features, 
                                    input_mean=hparams.mean,
                                    input_scale=hparams.std,
                                    max_bin=hparams.max_bin,
                                    sample_rate=hparams.sample_rate,
                                    add_emb=hparams.add_emb,
                                    )

    return model


def loss_creator(hparams):
    if hparams.loss_name == 'l1':
        loss_func = torch.nn.L1Loss()

    elif hparams.loss_name == 'mse':
        loss_func = torch.nn.MSELoss()

    return loss_func


def dataset_creator(hparams, partition):
    aug_list = []
    if hparams.aug_gain: aug_list.append('_augment_gain')
    if hparams.aug_channelswap: aug_list.append('_augment_channelswap')
    if hparams.aug_pitchShift: aug_list.append('_augment_pitchShift')
    source_augmentations = Compose(
            [globals()[aug] for aug in aug_list]
        )

    if hparams.dataset_name == 'musdb' or partition == 'test':
        dataset_kwargs = {
            'root': '../data/MUSDB18-HQ/', #hparams.data_path,
            'is_wav': True,
            'subsets': 'train' if partition!='test' else 'test',
            'target': hparams.target,
            'download': False,
            'seed': hparams.seed
        }

        dataset = MUSDBDataset(
            split=partition,
            samples_per_track=hparams.samples_per_track if partition=='train' else 1,
            seq_duration=hparams.seq_dur if partition=='train' else None,
            source_augmentations=source_augmentations if partition=='train' else None,
            random_track_mix=True if partition=='train' else False, add_emb = hparams.add_emb, 
            emb_feature=hparams.emb_feature,
            **dataset_kwargs
        )

    elif hparams.dataset_name == 'slakh':
        dataset = SlakhDataset(
            target=hparams.target,
            root=hparams.data_path,
            sf2_dir = hparams.sf2_dir,
            seq_duration=hparams.seq_dur,
            samples_per_track=hparams.samples_per_track,
            source_augmentations=source_augmentations if partition=='train' else None,
            seed=42,
            split = partition
        )

    return dataset