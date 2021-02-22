import argparse
import random
import glob
import os
import tqdm
import yaml 
import subprocess as sp

import numpy as np
import torch

from utils.general_utils import yaml_to_parser, seed_everything, save_checkpoint
from separator import Separator

def main(hparams, yaml_hparam):
    seed_everything()
    # data
    t = tqdm.trange(1, hparams.epochs + 1, disable=False)
    train_losses = []
    valid_losses = []
    best_epoch = 0

    ckpt_path = os.path.join('logger', hparams.model_name+'_'+hparams.dataset_name+'_'+hparams.emb_feature+'_conv64_lstm')
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)

    with open(os.path.join(ckpt_path, hparams.target+'.yaml'), 'w') as file:
        documents = yaml.dump(yaml_hparam, file)

    separator = Separator(hparams)
    if hparams.load_pretrain:
        checkpoint = torch.load(os.path.join(hparams.load_pretrain, hparams.target+'.pth')) 
        separator.model.load_state_dict(checkpoint)
    for epoch in t:
        t.set_description("Training Epoch")
        train_loss = separator.training_step()
        valid_loss = separator.validation_step()
        separator.scheduler.step(valid_loss)
        train_losses.append(train_loss)
        valid_losses.append(valid_loss)
        
        t.set_postfix(
            train_loss=train_loss, val_loss=valid_loss
        )
        print(valid_loss)
        stop = separator.es.step(valid_loss)

        if valid_loss == separator.es.best:
            best_epoch = epoch

        save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': separator.model.state_dict(),
                'best_loss': separator.es.best,
                'optimizer': separator.optimizer.state_dict(),
                'scheduler': separator.scheduler.state_dict()
            },
            is_best=valid_loss == separator.es.best,
            path=ckpt_path,
            target=hparams.target
        )

        if stop:
            print("Apply Early Stopping")
            break

    ffmpeg_command = f'python test.py --ckpt_path {ckpt_path} --target {hparams.target}'
    sp.call(ffmpeg_command, shell=True)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    args, other = parser.parse_known_args()
    hparam, yaml_hparam = yaml_to_parser('parameters.yaml')
    hparam = hparam.parse_args(other)
    main(hparam, yaml_hparam)