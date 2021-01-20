import pytorch_lightning as pl
import torch
import random
from loguru import logger
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import os
from utils.creator import dataset_creator, model_creator, preprocess_creator, loss_creator
import museval
from utils.general_utils import get_statistics, bandwidth_to_max_bin, EarlyStopping, AverageMeter
from utils.augmentation import _augment_freq_masking
import tqdm

#os.environ["CUDA_VISIBLE_DEVICES"] = "1"

class Separator(object):

    def __init__(self, hparams):
        super().__init__()

        self.hparams = hparams
        self.use_device = str("cuda" if torch.cuda.is_available() else "cpu")

        # dataset
        self.train_set, self.val_set = self._get_data_loader('train'), self._get_data_loader('valid')
        # preprocess
        self.transform = preprocess_creator(hparams)

        hparams.mean, hparams.std = self._get_mean_std()
        hparams.max_bin = bandwidth_to_max_bin(self.hparams.sample_rate, self.hparams.n_fft, self.hparams.band_width)

        self.model = model_creator(hparams).to(self.use_device)
        self.loss_func = loss_creator(hparams)

        self.optimizer, self.scheduler = self._configure_optimizers()
        self.es = EarlyStopping(patience=hparams.early_stopping_patience)

        self.transform.to(self.use_device)
        

    def forward(self, batch, partition):
        oup_dict = {}
        
        # preprocessing
        mix_audio, tar_audio = batch

        if self.hparams.dataset_name == 'slakh':
            mix_audio = mix_audio.permute(1, 0, 2).float()
            tar_audio = tar_audio.permute(1, 0, 2).float()

        batch_size = mix_audio.shape[0]
        if partition == 'va':
            mix_audio = mix_audio[..., :int(mix_audio.shape[-1] // 8) * 8]
            tar_audio = tar_audio[..., :int(mix_audio.shape[-1] // 8) * 8]
            mix_audio = mix_audio.permute(0, 2, 1).reshape(batch_size * 8, -1, self.hparams.n_channels).permute(0, 2, 1)
            tar_audio = tar_audio.permute(0, 2, 1).reshape(batch_size * 8, -1, self.hparams.n_channels).permute(0, 2, 1)
        mix_stft = self.transform(mix_audio)
        tar_stft = self.transform(tar_audio)
        mix_mag = mix_stft.pow(2).sum(-1).pow(1 / 2.0)
        tar_mag = tar_stft.pow(2).sum(-1).pow(1 / 2.0)
        # (batch, channel, n_features, n_frames)
        mix_mag_detach = mix_mag.detach().clone()

        if self.hparams.aug_freqmask:
            mix_mag = _augment_freq_masking(mix_mag)
        pre_mag = self.model(mix_mag, mix_mag_detach)

        loss = self.loss_func(pre_mag, tar_mag)

        return loss

    def training_step(self):
        losses = AverageMeter()
        self.model.train()
        pbar = tqdm.tqdm(self.train_set, disable=False)
        for x, y, _ in pbar:
            x, y = x.to(self.use_device), y.to(self.use_device)
            self.optimizer.zero_grad()
            loss = self.forward((x, y), 'tr')
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), x.shape[1])
        return losses.avg


    def validation_step(self):
        losses = AverageMeter()
        self.model.eval()
        pbar = tqdm.tqdm(self.val_set, disable=False)
        with torch.no_grad():
            for x, y, _ in pbar:
                x, y = x.to(self.use_device), y.to(self.use_device)
                loss = self.forward((x, y), 'va')
                losses.update(loss.item(), x.shape[1])
        return losses.avg

    def test_step(self, mix_audio, tar_audio, track_id):
        import norbert
        import scipy.signal
        import museval
        import numpy as np

        self.model.eval()

        mix_audio = mix_audio.to(self.use_device)
        mix_stft = self.transform(mix_audio)
        mix_mag = mix_stft.pow(2).sum(-1).pow(1 / 2.0)

        #pre_mag = self.model(mix_mag, mix_mag.detach().clone())
        pre_mag_list = []
        chunk_size = int(mix_mag.shape[-1] // 4) + 1

        for i in range(mix_mag.shape[-1] // chunk_size + 1):
            inp = mix_mag[..., i * chunk_size : (i + 1) * chunk_size]
            pre_mag_list.append(self.model(inp, inp.detach()))
        pre_mag = torch.cat(pre_mag_list, -1)

        pre_mag = pre_mag.squeeze().permute(2, 1, 0).detach().cpu().numpy()

        mix_stft = mix_stft.squeeze().detach().cpu().numpy()
        mix_stft = mix_stft[..., 0] + mix_stft[..., 1]*1j
        pre_residual = mix_mag.squeeze().permute(2, 1, 0).detach().cpu().numpy() - pre_mag
        pre_stft = np.concatenate((pre_mag[..., None], pre_residual[..., None]), -1) * np.exp(1j*np.angle(np.transpose(mix_stft, (2, 1, 0))[..., None]))
        #pre_stft = norbert.wiener(np.concatenate((pre_mag[..., None], pre_residual[..., None]), -1), 
        #                          mix_stft.T.astype(np.complex128), 1, use_softmask=False)
        print(pre_stft.shape)
        estimates = {}
        names = ['vocals', 'accompaniment']
        for i, name in enumerate(names):

            t, pre_audio = scipy.signal.istft(
                pre_stft[...,i].T / (self.hparams.n_fft / 2),
                44100,
                nperseg=self.hparams.n_fft,
                noverlap=self.hparams.n_fft - self.hparams.hop_length,
                boundary=True
            )
            estimates[name] = pre_audio.T

        track = self.loaders['test'].mus.tracks[track_id]
        scores = museval.eval_mus_track(track, estimates)
        print(scores)
        return scores


    def _configure_optimizers(self):

        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.hparams.learning_rate, weight_decay=self.hparams.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=self.hparams.patience, factor=self.hparams.decay_factor, cooldown=10, verbose=True)

        return optimizer, scheduler

    def _get_data_loader(self, partition):
        self.loaders = {}
        batch_size = self.hparams.batch_size if partition == 'train' else 1
        shuffle = True if partition == 'train' else False
        self.loaders[partition] = dataset_creator(self.hparams, partition)

        return data.DataLoader(self.loaders[partition], batch_size, shuffle=shuffle, num_workers=self.hparams.num_workers, pin_memory=True)

    def _get_mean_std(self):
        if self.hparams.use_norm:
            mean, std = get_statistics(self.hparams, dataset_creator(self.hparams, 'train'), self.transform.cpu())
        else:
            mean, std = None, None

        return mean, std

    @classmethod
    def load_from_checkpoint(cls, checkpoint_path, default_params):
        checkpoint = torch.load(checkpoint_path, map_location=lambda storage, loc: storage)
        import yaml
        from argparse import Namespace
        default_params = yaml.safe_load(open(default_params))
        inp_harams = Namespace(**default_params)
        inp_harams.use_norm = False

        separator = cls(inp_harams)
        pretrained_dict = checkpoint
        separator.model.load_state_dict(pretrained_dict)

        separator.transform.center = True

        return separator