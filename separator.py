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
import sys
import time

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

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
        mix_audio, tar_audio = mix_audio.float(), tar_audio.float()
        if mix_audio.shape[1] != self.hparams.n_channels:
            mix_audio = (mix_audio.sum(1) / mix_audio.shape[1]).unsqueeze(1)
            tar_audio = (tar_audio.sum(1) / tar_audio.shape[1]).unsqueeze(1)
        #if self.hparams.dataset_name == 'slakh':
        #    mix_audio = mix_audio.permute(1, 0, 2)
        #    tar_audio = tar_audio.permute(1, 0, 2)

        batch_size = mix_audio.shape[0]
        mix_stft, mix_mag = self.transform(mix_audio)
        tar_stft, tar_mag = self.transform(tar_audio)
        
        if partition == 'va':
            chunk_size = mix_mag.shape[-1]//2
            oup_list = []
            for i in range(2):
                # (batch, channel, n_features, n_frames)
                mix_mag_detach = mix_mag[..., i*chunk_size:(i+1)*chunk_size].detach().clone()
                pre_mag = self.model(mix_mag[..., i*chunk_size:(i+1)*chunk_size], mix_mag_detach)
                oup_list.append(pre_mag)
            pre_mag = torch.cat(oup_list, -1)
            tar_mag = tar_mag[..., :pre_mag.shape[-1]]
        else:
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
        st = time.time()
        #pbar = tqdm.tqdm(self.train_set, disable=False)
        for i, (x, y, _) in enumerate(self.train_set):
            x, y = x.to(self.use_device), y.to(self.use_device)
            self.optimizer.zero_grad()
            loss = self.forward((x, y), 'tr')
            loss.backward()
            self.optimizer.step()
            losses.update(loss.item(), x.shape[1])

            sys.stdout.write('\r')
            sys.stdout.write('| Iter[%4d/%4d]\tLoss %4f\tTime %d'%(i+1, len(self.train_set), loss.item(), time.time() - st))
            sys.stdout.flush()

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
        # preprocessing feature
        audio_torch = mix_audio.float().to(self.use_device)

        source_names = [self.hparams.target, 'accompaniment']
        oup_list = []        
        X_stft, X_mag = self.transform(audio_torch)
        chunk_size = X_mag.shape[-1]//3
        for i in range(3):
            pre_mag = self.model(X_mag[..., i*chunk_size:(i+1)*chunk_size], X_mag[..., i*chunk_size:(i+1)*chunk_size].detach().clone()).cpu().detach().numpy()
            oup_list.append(pre_mag)
        pre_mag = np.concatenate(oup_list, -1)
        X_stft = X_stft[:, :, :, :pre_mag.shape[-1]]
        # output is nb_samples, nb_channels, nb_bins, nb_frames

        pre_mag = np.transpose(pre_mag, (3, 2, 1, 0))

        X_stft = X_stft.detach().cpu().numpy()
        X_stft = X_stft[..., 0] + X_stft[..., 1]*1j
        X_stft = X_stft[0].transpose(2, 1, 0)
        pre_mag = np.concatenate((pre_mag, np.abs(X_stft)[..., None] - pre_mag), axis=-1)
        # frames, bins, channels, sources
        pre_stft = pre_mag * np.exp(1j*np.angle(X_stft[..., None]))

        audio_estimates = []
        for j, name in enumerate(source_names):
            _, audio_hat = scipy.signal.istft(
                pre_stft[..., j].T / (self.hparams.n_fft / 2),
                44100,
                nperseg=self.hparams.n_fft,
                noverlap=self.hparams.n_fft - self.hparams.hop_length,
                boundary=True
            )
            audio_estimates.append(audio_hat.T)


        # gather reference tracks
        track = self.loaders['test'].mus.tracks[track_id]
        ref_tar = track.targets[self.hparams.target].audio
        ref_res = track.audio - track.targets[self.hparams.target].audio
        if ref_tar.shape[1] != self.hparams.n_channels:
            ref_tar = (ref_tar.sum(1) / ref_tar.shape[1])[..., None]
            ref_res = (ref_res.sum(1) / ref_res.shape[1])[..., None]
        audio_reference = [ref_tar, ref_res]
        #scores = museval.eval_mus_track(track, audio_estimates)

        SDR, ISR, SIR, SAR = museval.evaluate(
            audio_reference,
            audio_estimates,
            win=int(1.0*44100),
            hop=int(1.0*44100),
            mode='v4'
        )
        scores = museval.aggregate.TrackStore(win=1.0, hop=1.0, track_name=track.name)
        for i, target in enumerate(source_names):
            values = {
                "SDR": SDR[i].tolist(),
                "SIR": SIR[i].tolist(),
                "ISR": ISR[i].tolist(),
                "SAR": SAR[i].tolist()
            }

            scores.add_target(
                target_name=target,
                values=values
            )
        
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
        checkpoint = torch.load(checkpoint_path) #, map_location=lambda storage, loc: storage)
        import yaml
        from argparse import Namespace
        default_params = yaml.safe_load(open(default_params))
        inp_harams = Namespace(**default_params)
        inp_harams.use_norm = False

        separator = cls(inp_harams)

        model_dict = separator.model.state_dict()
        pretrained_dict = {k: v for k, v in checkpoint.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        separator.model.load_state_dict(model_dict)

        separator.transform.center = True

        return separator