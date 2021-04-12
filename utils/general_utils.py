import yaml
import os
import argparse
import random
import numpy as np
from pytorch_lightning.callbacks import Callback
import torch
from pytorch_lightning.callbacks import EarlyStopping
from loguru import logger
from pytorch_lightning.callbacks import ModelCheckpoint

# turn "None" string into None
def _strings_handle_none(arg):
    if arg.lower() in ['null', 'none']:
        return None
    else:
        return str(arg)

# turn "True/False" string into True/False
def _bool_string_to_bool(arg):
    if str(arg).lower() == "false":
        return False
    if str(arg).lower() == 'true':
        return True

# parse yaml format into parameters
def yaml_to_parser(yaml_path):
    parser = argparse.ArgumentParser()
    hparams = yaml.safe_load(open(yaml_path))
    
    for k, val in hparams.items():
        if isinstance(val, str):
            argparse_kwargs = {'type': _strings_handle_none, 'default': val}
        elif isinstance(val, bool):
            argparse_kwargs = {'type': _bool_string_to_bool, 'default': val}
        else:
            argparse_kwargs = {'type': eval, 'default': val}

        parser.add_argument('--{}'.format(k.replace('_', '-')), **argparse_kwargs)
    return parser, hparams

# fix the random seed
def seed_everything(seed=42):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    #torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    #torch.backends.cudnn.enabled = False 


class EarlyStopping(object):
    def __init__(self, mode='min', min_delta=0, patience=10):
        self.mode = mode
        self.min_delta = min_delta
        self.patience = patience
        self.best = None
        self.num_bad_epochs = 0
        self.is_better = None
        self._init_is_better(mode, min_delta)

        if patience == 0:
            self.is_better = lambda a, b: True

    def step(self, metrics):
        if self.best is None:
            self.best = metrics
            return False

        if np.isnan(metrics):
            return True

        if self.is_better(metrics, self.best):
            self.num_bad_epochs = 0
            self.best = metrics
        else:
            self.num_bad_epochs += 1

        if self.num_bad_epochs >= self.patience:
            return True

        return False

    def _init_is_better(self, mode, min_delta):
        if mode not in {'min', 'max'}:
            raise ValueError('mode ' + mode + ' is unknown!')
        if mode == 'min':
            self.is_better = lambda a, best: a < best - min_delta
        if mode == 'max':
            self.is_better = lambda a, best: a > best + min_delta

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def get_statistics(args, dataset, preprocessor):
    import sklearn
    import numpy as np
    import copy
    import tqdm

    scaler = sklearn.preprocessing.StandardScaler()

    dataset_scaler = copy.deepcopy(dataset)
    dataset_scaler.samples_per_track = 1
    dataset_scaler.augmentations = None
    dataset_scaler.random_track_mix = False
    dataset_scaler.seq_duration = None
    pbar = tqdm.tqdm(range(len(dataset_scaler)))
    for ind in pbar:
        x, y, _ = dataset_scaler[ind]
        pbar.set_description("Compute dataset statistics")
        X = preprocessor((x.sum(0)/2)[None, None, ...])
        X = X.pow(2).sum(-1).pow(1 / 2.0).permute(3, 0, 1, 2).detach().cpu().numpy()
        scaler.partial_fit(np.squeeze(X))

    # set inital input scaler values
    std = np.maximum(
        scaler.scale_,
        1e-4*np.max(scaler.scale_)
    )

    return scaler.mean_, std

def bandwidth_to_max_bin(rate, n_fft, bandwidth):
    freqs = np.linspace(
        0, float(rate) / 2, n_fft // 2 + 1,
        endpoint=True
    )

    return np.max(np.where(freqs <= bandwidth)[0]) + 1

def save_checkpoint(state, is_best, path, target):
    # save full checkpoint including optimizer
    if is_best:
        # save just the weights
        torch.save(
            state['state_dict'],
            os.path.join(path, target + '.pth')
        )

def target_to_midi_number(target):
    if target == 'bass':
        numbers = np.arange(33, 40)

    return numbers

def sdr_loss_core(input, gt, mix, weighted=True):
    assert input.shape == gt.shape # (Batch, Len)
    assert mix.shape == gt.shape   # (Batch, Len)

    input = input[:, 200:-200]
    gt = gt[:, 200:-200]
    mix = mix[:, 200:-200]

    ns = mix - gt
    ns_hat = mix - input

    if weighted:
        alpha = torch.sum((gt*gt), 1, keepdims=True) / (torch.sum((gt*gt), 1,
                                                          keepdims=True) + torch.sum((ns*ns), 1, keepdims=True) + 1e-10)
    else:
        alpha = 0.5

    # Target
    num_cln = torch.sum((input*gt), 1, keepdims=True)
    denom_cln = ((1e-10 + torch.sum((input*input), 1, keepdims=True))
                 ** 0.5) * ((1e-10 + torch.sum((gt*gt), 1, keepdims=True)) ** 0.5)
    sdr_cln = num_cln / (denom_cln + 1e-10)

    # Noise
    num_noise = torch.sum((ns*ns_hat), 1, keepdims=True)
    denom_noise = ((1e-10 + torch.sum((ns_hat*ns_hat), 1, keepdims=True))
                   ** 0.5) * ((1e-10 + torch.sum((ns*ns), 1, keepdims=True)) ** 0.5)
    sdr_noise = num_noise / (denom_noise + 1e-10)

    return torch.mean(-alpha*sdr_cln - (1. - alpha)*sdr_noise)