import pytorch_lightning as pl
import torch
import random
from loguru import logger
import numpy as np
import torch.utils.data as data
import torch.nn as nn
import torch.nn.functional as F
import os
from utils.creator import dataset_creator, model_creator, preprocess_creator, loss_creator
import museval
from utils.general_utils import get_statistics, bandwidth_to_max_bin, EarlyStopping, AverageMeter
from utils.augmentation import _augment_freq_masking
import tqdm
import soundfile as sf
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

        self.model = model_creator(hparams, self.use_device).to(self.use_device)
   
        self.loss_func = loss_creator(hparams.loss_name)
        #if self.hparams.model_name == 'x_umix':
        #    self.sdr_loss = loss_creator('sdr')

        self.optimizer, self.scheduler = self._configure_optimizers()
        self.es = EarlyStopping(patience=hparams.early_stopping_patience)

        self.transform.to(self.use_device)

        #if self.hparams.use_emb:
        self.emb_loss = loss_creator('cosEmb') # other
        #self.emb_loss_2 = loss_creator('cosEmb', 0.1)


    def forward(self, batch, partition):
        oup_dict = {}
        loss = 0
        
        # preprocessing
        if partition == 'va':
            mix_audio, tar_audio = batch
            batch_size = mix_audio.shape[0]
            c_size = 8
            mix_audio = mix_audio[..., :int(mix_audio.shape[-1] // c_size) * c_size]
            tar_audio = tar_audio[..., :int(tar_audio.shape[-1] // c_size) * c_size] #batch, sources, channel, time
            mix_audio = mix_audio.permute(0, 2, 1).reshape(batch_size * c_size, -1, 2).permute(0, 2, 1)
            tar_audio = tar_audio.permute(0, 2, 1).reshape(batch_size * c_size, -1, 2).permute(0, 2, 1)
            vgg = None
        else:
            mix_audio, tar_audio, vgg = batch
            vgg = vgg.repeat(1, 1, 43, 1)[:, :, 1:-2].permute(1, 0, 2, 3).reshape(4, -1, 128)
        if len(tar_audio.shape) == 4:
            tar_audio = tar_audio[:,:].reshape(-1, tar_audio.shape[2], tar_audio.shape[3])
        mix_stft = self.transform(mix_audio)
        tar_stft = self.transform(tar_audio)
        mix_mag = mix_stft.pow(2).sum(-1).pow(1 / 2.0)
        tar_mag = tar_stft.pow(2).sum(-1).pow(1 / 2.0)
        # (batch, channel, n_features, n_frames)
        mix_mag_detach = mix_mag.detach().clone()

        if self.hparams.aug_freqmask:
            mix_mag = _augment_freq_masking(mix_mag)

        if self.hparams.use_emb and partition=='tr':
            pre_mag, emb = self.model(mix_mag, mix_mag_detach)
            emb_loss_neg = 0
            emb_loss_pos = 0
            feature_num = emb[0].shape[0]

            # different group don't close to each other
            emb_loss_neg += torch.mean(self.emb_loss(emb[0], emb[1], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            emb_loss_neg += torch.mean(self.emb_loss(emb[1], emb[2], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            emb_loss_neg += torch.mean(self.emb_loss(emb[2], emb[3], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            emb_loss_neg += torch.mean(self.emb_loss(emb[0], emb[3], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            emb_loss_neg += torch.mean(self.emb_loss(emb[0], emb[2], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            emb_loss_neg += torch.mean(self.emb_loss(emb[1], emb[3], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            '''
            # same group close to each other
            for i in range(4):
                emb_loss_pos += torch.mean(self.emb_loss(emb[i][:int(feature_num//2)], emb[i][int(feature_num//2):], torch.ones(int(feature_num//2)).to(mix_mag.device)).view(feature_num//2, -1).sum(dim=-1))
            
            
            # same group close to vgg
            for i in range(4):
                emb_loss_pos += 0.01 * torch.mean(self.emb_loss(emb[i], vgg[i]/225, torch.ones(feature_num).to(self.use_device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            
            # different group don't close to vgg
            emb_loss_neg += torch.mean(self.emb_loss(emb[0], vgg[1], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            emb_loss_neg += torch.mean(self.emb_loss(emb[1], vgg[2], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            emb_loss_neg += torch.mean(self.emb_loss(emb[2], vgg[3], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            emb_loss_neg += torch.mean(self.emb_loss(emb[0], vgg[3], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            emb_loss_neg += torch.mean(self.emb_loss(emb[0], vgg[2], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            emb_loss_neg += torch.mean(self.emb_loss(emb[1], vgg[3], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
            '''

            loss += (emb_loss_pos + emb_loss_neg * 2)
            loss += self.loss_func(pre_mag, tar_mag)
            return (loss, emb_loss_pos, emb_loss_neg)

        else:
            pre_mag, emb = self.model(mix_mag, mix_mag_detach, vgg)
            emb_loss_neg = 0
            emb_loss_pos = 0
            if self.hparams.model_name == 'x_umix':
                if partition == 'tr':
                    tar_mag = tar_mag.reshape(pre_mag.shape)
                    loss = [self.loss_func(pre_mag[:,0], tar_mag[:,0]),
                             self.loss_func(pre_mag[:,1], tar_mag[:,1]),
                             self.loss_func(pre_mag[:,2], tar_mag[:,2]),
                             self.loss_func(pre_mag[:,3], tar_mag[:,3])]

                    '''
                    emb = [e.reshape(-1, e.shape[-1]) for e in emb]
                    feature_num = emb[0].shape[0]
                    emb_loss_neg += torch.mean(self.emb_loss(emb[0], emb[1], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    emb_loss_neg += torch.mean(self.emb_loss(emb[1], emb[2], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    emb_loss_neg += torch.mean(self.emb_loss(emb[0], emb[2], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    emb_loss_neg += torch.mean(self.emb_loss(emb[2], emb[3], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    emb_loss_neg += torch.mean(self.emb_loss(emb[0], emb[3], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    emb_loss_neg += torch.mean(self.emb_loss(emb[1], emb[3], -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    #emb_loss_neg += torch.mean(F.cosine_embedding_loss(emb[2], emb[3], -torch.ones(feature_num).to(mix_mag.device), 0, reduction='none').view(mix_mag.shape[0], -1).sum(dim=-1))
                    #emb_loss_neg += torch.mean(F.cosine_embedding_loss(emb[0], emb[3], -torch.ones(feature_num).to(mix_mag.device), 0, reduction='none').view(mix_mag.shape[0], -1).sum(dim=-1))
                    #emb_loss_neg += torch.mean(F.cosine_embedding_loss(emb[1], emb[3], -torch.ones(feature_num).to(mix_mag.device), 0, reduction='none').view(mix_mag.shape[0], -1).sum(dim=-1))
                    #print(emb_loss_neg)
                    '''
                    #emb_loss_neg = emb_loss_neg * 0.01
                    '''
                    emb_loss_neg += torch.mean(self.emb_loss(emb[0], vgg[1]/225, -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    emb_loss_neg += torch.mean(self.emb_loss(emb[1], vgg[2]/225, -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    emb_loss_neg += torch.mean(self.emb_loss(emb[2], vgg[3]/225, -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    emb_loss_neg += torch.mean(self.emb_loss(emb[0], vgg[3]/225, -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    emb_loss_neg += torch.mean(self.emb_loss(emb[0], vgg[2]/225, -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    emb_loss_neg += torch.mean(self.emb_loss(emb[1], vgg[3]/225, -torch.ones(feature_num).to(mix_mag.device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                    
                    emb = [e.reshape(-1, e.shape[-1]) for e in emb]
                    feature_num = emb[0].shape[0]
                    for i in range(4):
                        emb_loss_pos += 0.0001 * torch.mean(self.emb_loss(emb[i], vgg[i]/225, torch.ones(feature_num).to(self.use_device)).view(mix_mag.shape[0], -1).sum(dim=-1))
                  
                    '''
                    #loss.append(emb_loss_neg * 0.01)
                    
                else:
                    loss = self.loss_func(pre_mag[:,0], tar_mag)
                '''
                x_theta = torch.atan2(mix_stft[..., 0], mix_stft[..., 1])
                real = torch.cos(x_theta).unsqueeze(1) * pre_mag
                img = torch.sin(x_theta).unsqueeze(1) * pre_mag
                _, _, _, n_bin, n_frame = real.shape
                pred = torch.istft(torch.cat((real.reshape(-1, n_bin, n_frame).unsqueeze(-1), img.reshape(-1, n_bin, n_frame).unsqueeze(-1)), -1), self.hparams.n_fft, self.hparams.hop_length, window=torch.hann_window(self.hparams.n_fft).to(self.use_device))
                length = pred.shape[-1]
                sdr_loss = 1 + self.sdr_loss(pred, tar_audio.reshape(-1, tar_audio.shape[-1])[..., :length], mix_audio.unsqueeze(1).repeat(1, 4, 1, 1).reshape(-1, mix_audio.shape[-1])[..., :length])
                loss.append(sdr_loss*10)
                '''
            else:
                loss += self.loss_func(pre_mag, tar_mag)
            
            return loss

    def training_step(self):
        losses = AverageMeter()
        #emb_pos_losses = AverageMeter()
        #emb_neg_losses = AverageMeter()
        st = time.time()
        self.model.train()
        pbar = tqdm.tqdm(self.train_set, disable=False)
        for x, y, _, vgg in pbar:
            x, y, vgg = x.to(self.use_device), y.to(self.use_device), vgg.to(self.use_device)
            self.optimizer.zero_grad()
            #loss, emb_loss_pos, emb_loss_neg = self.forward((x, y, vgg), 'tr')
            loss = self.forward((x, y, vgg), 'tr')
            total_loss = sum(loss[:4]) / 4 #+ loss[-1]
            total_loss.backward()

            self.optimizer.step()
            losses.update(total_loss.item(), x.shape[1])
            #emb_pos_losses.update(emb_loss_pos.item(), x.shape[1])
            #emb_neg_losses.update(emb_loss_neg.item(), x.shape[1])
            pbar.set_postfix({'vocals': loss[0].item(), 'drums': loss[1].item(), 'bass': loss[2].item(), 'other': loss[3].item()})
        return losses.avg#, emb_pos_losses.avg, emb_neg_losses.avg


    def validation_step(self):
        losses = AverageMeter()
        self.model.eval()
        with torch.no_grad():
            for x, y, _ in self.val_set:
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
        self.model = self.model.to(self.use_device)

        mix_stft = self.transform(mix_audio)
        mix_mag = mix_stft.pow(2).sum(-1).pow(1 / 2.0)

        #pre_mag = self.model(mix_mag, mix_mag.detach().clone())
        pre_mag_list = []
        chunk_size = int(mix_mag.shape[-1] // 2) + 1
        for i in range(mix_mag.shape[-1] // chunk_size + 1):
            inp = mix_mag[..., i * chunk_size : (i + 1) * chunk_size].to(self.use_device)
            oup, emb = self.model(inp, inp.detach(), inp)
            pre_mag_list.append(oup.squeeze().detach().cpu().numpy())
        
            '''
            np.save(f'tmp_ori/0_{str(i)}_{str(track_id.detach().cpu().numpy()[0])}.npy', emb[0].detach().cpu().numpy())
            np.save(f'tmp_emb/1_{str(i)}_{str(track_id.detach().cpu().numpy()[0])}.npy', emb[1].detach().cpu().numpy())
            np.save(f'tmp_emb/2_{str(i)}_{str(track_id.detach().cpu().numpy()[0])}.npy', emb[2].detach().cpu().numpy())
            np.save(f'tmp_emb/3_{str(i)}_{str(track_id.detach().cpu().numpy()[0])}.npy', emb[3].detach().cpu().numpy())
        return None
        '''
        pre_mag = np.concatenate(pre_mag_list, -1).T

        mix_stft = mix_stft.squeeze().detach().cpu().numpy()
        mix_stft = mix_stft[..., 0] + mix_stft[..., 1]*1j
        pre_stft = norbert.wiener(pre_mag, mix_stft.T.astype(np.complex128), 1, use_softmask=False)
        #pre_residual = mix_mag.squeeze().permute(2, 1, 0).detach().cpu().numpy() - pre_mag
        #pre_stft = np.concatenate((pre_mag[..., None], pre_residual[..., None]), -1) * np.exp(1j*np.angle(np.transpose(mix_stft, (2, 1, 0))[..., None]))
        #pre_stft = norbert.wiener(np.concatenate((pre_mag[..., None], pre_residual[..., None]), -1), 
        #                          mix_stft.T.astype(np.complex128), 1, use_softmask=False)
        estimates = {}
        #names = ['vocals', 'accompaniment']
        names = ['vocals', 'drums', 'bass', 'other']
        for i, name in enumerate(names):

            t, pre_audio = scipy.signal.istft(
                pre_stft[...,i].T / (self.hparams.n_fft / 2),
                44100,
                nperseg=self.hparams.n_fft,
                noverlap=self.hparams.n_fft - self.hparams.hop_length,
                boundary=True
            )
            estimates[name] = pre_audio.T

        #sf.write(f'{track_id}_vocals.wav', estimates['vocals'], 44100)
        track = self.loaders['test'].mus.tracks[track_id]
        scores = museval.eval_mus_track(track, estimates)
        print(scores)
        return scores
        #return None
        

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

        model_dict = separator.model.state_dict()
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(pretrained_dict) 
        separator.model.load_state_dict(model_dict)

        separator.transform.center = True

        return separator
