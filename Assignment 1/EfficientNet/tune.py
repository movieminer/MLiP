# %%
import albumentations as A
import gc
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import os
from pathlib import Path
import pandas as pd
import random
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from torch.optim import AdamW
from sklearn.model_selection import train_test_split, KFold, GroupKFold
import tempfile

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List

import ray
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

tempfile.tempdir = '/home/jdusseljee/efficientnet/tmp'
os.environ['TUNE_RESULT_DIR'] = '/home/jdusseljee/efficientnet/results'

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

# %% [markdown]
# # Config

# %%
class config:
    DEVICE = 'cuda'
    AMP = True
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VALID = 32
    EPOCHS = 10
    FOLDS = 5
    VAL_SIZE = .2   # If FOLDS=1 
    FREEZE = False
    MAX_GRAD_NORM = 1e7
    MODEL = "efficientnet_b0"
    PRETRAINED = True
    NUM_FROZEN_LAYERS = 39
    NUM_WORKERS = 0 # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    TRAIN_FULL_DATA = False
    VISUALIZE = True
    WEIGHT_DECAY = 0.1
    
    
class paths:
    OUTPUT_DIR = "/home/jdusseljee/efficientnet/working/"
    DATASET = '/home/jdusseljee/efficientnet/input/dataset/'

# %% [markdown]
# # Utility functions

# %%
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


def asMinutes(s: float):
    "Convert to minutes."
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)


def timeSince(since: float, percent: float):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (remain %s)' % (asMinutes(s), asMinutes(rs))


def get_logger(filename=paths.OUTPUT_DIR):
    from logging import getLogger, INFO, StreamHandler, FileHandler, Formatter
    logger = getLogger(__name__)
    logger.setLevel(INFO)
    handler1 = StreamHandler()
    handler1.setFormatter(Formatter("%(message)s"))
    handler2 = FileHandler(filename=f"{filename}.log")
    handler2.setFormatter(Formatter("%(message)s"))
    logger.addHandler(handler1)
    logger.addHandler(handler2)
    return logger
    
    
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 

    
def sep():
    print("-"*100)
    

label_cols = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
target_preds = [x + "_pred" for x in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other':5}
num_to_label = {v: k for k, v in label_to_num.items()}
LOGGER = get_logger()
seed_everything(config.SEED)

# %% [markdown]
# # Dataset

# %%
class EEGDataset(Dataset):
    def __init__(self, train_df, kaggle_spectrograms, eeg_spectrograms, mode='train', augment='both'):
        self.train_df = train_df
        self.kaggle_specs = kaggle_spectrograms
        self.eeg_specs = eeg_spectrograms
        self.mode = mode
        self.augment = augment

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        row = self.train_df.iloc[idx]

        X = np.zeros((128,256,8),dtype='float32')
        img = np.ones((128,256),dtype='float32')

        kaggle_spec = self.kaggle_specs[row['spectrogram_id']]
        eeg_spec = self.eeg_specs[row['eeg_id']]

        if self.mode=='test': 
            r = 0
        else: 
            r = int((row['min'] + row['max']) // 4)

        for k in range(4):
            # EXTRACT 300 ROWS OF SPECTROGRAM
            img = kaggle_spec[r:r+300,k*100:(k+1)*100].T

            # LOG TRANSFORM SPECTROGRAM
            img = np.clip(img,np.exp(-4),np.exp(8))
            img = np.log(img)

            # STANDARDIZE PER IMAGE
            ep = 1e-6
            m = np.nanmean(img.flatten())
            s = np.nanstd(img.flatten())
            img = (img-m)/(s+ep)
            img = np.nan_to_num(img, nan=0.0)

            # CROP TO 256 TIME STEPS
            X[14:-14,:,k] = img[:,22:-22] / 2.0

        X[:,:,4:8] = eeg_spec

        X = self.__transform(X)
        
        if self.mode != 'test':
            y = torch.from_numpy(row[label_cols].values.astype(np.float32))

        return torch.from_numpy(X), y

    def __transform(self, img):
        params1 = {
                    "num_masks_x": 1,    
                    "mask_x_length": (0, 20), # This line changed from fixed  to a range
                    "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),
                    }
        params2 = {    
                    "num_masks_y": 1,    
                    "mask_y_length": (0, 20),
                    "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),    
                    }
        params3 = {    
                    "num_masks_x": (2, 4),
                    "num_masks_y": 5,    
                    "mask_y_length": 8,
                    "mask_x_length": (10, 20),
                    "fill_value": (0, 1, 2, 3, 4, 5, 6, 7),  
                    }

        if self.augment == 'none':
            return img
        elif self.augment == 'flip':
            transforms = A.Compose([
                A.HorizontalFlip(p=0.5),
            ])
        elif self.augment == 'mask':
            transforms = A.Compose([
                A.XYMasking(**params1, p=0.3),
                A.XYMasking(**params2, p=0.3),
                A.XYMasking(**params3, p=0.3),
            ])
        elif self.augment == 'both':
            transforms = A.Compose([
                A.XYMasking(**params1, p=0.3),
                A.XYMasking(**params2, p=0.3),
                A.XYMasking(**params3, p=0.3),
                A.HorizontalFlip(p=0.5),
            ])
        return transforms(image=img)['image']
# %% [markdown]
# # Model

# %%
class ReshapeInput(nn.Module):
    def __init__(self, cfg):
        super(ReshapeInput, self).__init__()
        self.cfg = cfg

    def forward(self, x):
        # Split the 8-channel input into individual channels
        channels = [x[:, :, :, i:i+1] for i in range(8)]
        
        # Concatenate the first 4 channels along the height dimension
        x1 = torch.cat(channels[:4], dim=1)
        x2 = torch.cat(channels[4:], dim=1)
        x3 = torch.cat([x1,x2], dim=2)
        
        # Concatenate the channels along the width dimension to form a 3-channel 512x512 image
        x = torch.cat([x3] if self.cfg['channels'] == 1 else [x3, x3, x3], dim=3)  # Stack the same image thrice along the channel dimension
        x = x.permute(0, 3, 1, 2)

        return x

class EEGNet(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.reshape = ReshapeInput(cfg=cfg)
        self.model = timm.create_model(
            model_name=cfg['model_name'], pretrained=config.PRETRAINED,
            num_classes=6, in_chans=cfg['channels'], drop_rate=cfg['drop_rate'], drop_path_rate=cfg['drop_path_rate'])

    def forward(self, x):
        x = self.reshape(x)
        x = self.model(x)      

        return x

# %% [markdown]
# ## Loss function

# %%
class KLDivLossWithLogits(nn.KLDivLoss):
    def __init__(self, reduction):
        super().__init__(reduction=reduction)

    def forward(self, y, t):
        y = F.log_softmax(y,  dim=1)
        loss = super().forward(y, t)

        return loss


import math
import torch
from torch.optim.optimizer import Optimizer


class Adan(Optimizer):
    """
    Implements a pytorch variant of Adan
    Adan was proposed in
    Adan: Adaptive Nesterov Momentum Algorithm for Faster Optimizing Deep Models[J]. arXiv preprint arXiv:2208.06677, 2022.
    https://arxiv.org/abs/2208.06677
    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining parameter groups.
        lr (float, optional): learning rate. (default: 1e-3)
        betas (Tuple[float, float, flot], optional): coefficients used for computing 
            running averages of gradient and its norm. (default: (0.98, 0.92, 0.99))
        eps (float, optional): term added to the denominator to improve 
            numerical stability. (default: 1e-8)
        weight_decay (float, optional): decoupled weight decay (L2 penalty) (default: 0)
        max_grad_norm (float, optional): value used to clip 
            global grad norm (default: 0.0 no clip)
        no_prox (bool): how to perform the decoupled weight decay (default: False)
    """

    def __init__(self, params, lr=1e-3, betas=(0.98, 0.92, 0.99), eps=1e-8,
                 weight_decay=0.2, max_grad_norm=0.0, no_prox=False):
        if not 0.0 <= max_grad_norm:
            raise ValueError("Invalid Max grad norm: {}".format(max_grad_norm))
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError("Invalid beta parameter at index 0: {}".format(betas[0]))
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError("Invalid beta parameter at index 1: {}".format(betas[1]))
        if not 0.0 <= betas[2] < 1.0:
            raise ValueError("Invalid beta parameter at index 2: {}".format(betas[2]))
        defaults = dict(lr=lr, betas=betas, eps=eps,
                        weight_decay=weight_decay,
                        max_grad_norm=max_grad_norm, no_prox=no_prox)
        super(Adan, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(Adan, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('no_prox', False)

    @torch.no_grad()
    def restart_opt(self):
        for group in self.param_groups:
            group['step'] = 0
            for p in group['params']:
                if p.requires_grad:
                    state = self.state[p]
                    # State initialization

                    # Exponential moving average of gradient values
                    state['exp_avg'] = torch.zeros_like(p)
                    # Exponential moving average of squared gradient values
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    # Exponential moving average of gradient difference
                    state['exp_avg_diff'] = torch.zeros_like(p)

    @torch.no_grad()
    def step(self):
        """
            Performs a single optimization step.
        """
        if self.defaults['max_grad_norm'] > 0:
            device = self.param_groups[0]['params'][0].device
            global_grad_norm = torch.zeros(1, device=device)

            max_grad_norm = torch.tensor(self.defaults['max_grad_norm'], device=device)
            for group in self.param_groups:

                for p in group['params']:
                    if p.grad is not None:
                        grad = p.grad
                        global_grad_norm.add_(grad.pow(2).sum())

            global_grad_norm = torch.sqrt(global_grad_norm)

            clip_global_grad_norm = torch.clamp(max_grad_norm / (global_grad_norm + group['eps']), max=1.0)
        else:
            clip_global_grad_norm = 1.0

        for group in self.param_groups:
            beta1, beta2, beta3 = group['betas']
            # assume same step across group now to simplify things
            # per parameter step can be easily support by making it tensor, or pass list into kernel
            if 'step' in group:
                group['step'] += 1
            else:
                group['step'] = 1

            bias_correction1 = 1.0 - beta1 ** group['step']

            bias_correction2 = 1.0 - beta2 ** group['step']

            bias_correction3 = 1.0 - beta3 ** group['step']

            for p in group['params']:
                if p.grad is None:
                    continue

                state = self.state[p]
                if len(state) == 0:
                    state['exp_avg'] = torch.zeros_like(p)
                    state['exp_avg_sq'] = torch.zeros_like(p)
                    state['exp_avg_diff'] = torch.zeros_like(p)

                grad = p.grad.mul_(clip_global_grad_norm)
                if 'pre_grad' not in state or group['step'] == 1:
                    state['pre_grad'] = grad

                copy_grad = grad.clone()

                exp_avg, exp_avg_sq, exp_avg_diff = state['exp_avg'], state['exp_avg_sq'], state['exp_avg_diff']
                diff = grad - state['pre_grad']

                update = grad + beta2 * diff
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)  # m_t
                exp_avg_diff.mul_(beta2).add_(diff, alpha=1 - beta2)  # diff_t
                exp_avg_sq.mul_(beta3).addcmul_(update, update, value=1 - beta3)  # n_t

                denom = ((exp_avg_sq).sqrt() / math.sqrt(bias_correction3)).add_(group['eps'])
                update = ((exp_avg / bias_correction1 + beta2 * exp_avg_diff / bias_correction2)).div_(denom)

                if group['no_prox']:
                    p.data.mul_(1 - group['lr'] * group['weight_decay'])
                    p.add_(update, alpha=-group['lr'])
                else:
                    p.add_(update, alpha=-group['lr'])
                    p.data.div_(1 + group['lr'] * group['weight_decay'])

                state['pre_grad'] = copy_grad

# %% [markdown]
# ## Train and Val functions

# %%
def train_epoch(train_loader, model, optimizer, epoch, scheduler):
    device = config.DEVICE
    model.train() 
    loss_func = KLDivLossWithLogits(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
    losses = AverageMeter()
    global_step = 0
    
    # with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
    for (X, y) in train_loader:
        X = X.to(device)
        y = y.to(device)
        batch_size = y.size(0)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=config.AMP):
            y_preds = model(X) 
            loss = loss_func(y_preds, y)
        losses.update(loss.item(), batch_size)
        
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    return losses.avg


def valid_epoch(valid_loader, model):
    device = config.DEVICE
    model.eval()
    loss_func = KLDivLossWithLogits(reduction="batchmean")
    softmax = nn.Softmax(dim=1)
    losses = AverageMeter()
    preds = np.empty((0,6))
    start = end = time.time()
    # with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
    for (X, y) in valid_loader:
        X = X.to(device)
        y = y.to(device)
        batch_size = y.size(0)
        with torch.no_grad():
            y_preds = model(X)
            loss = loss_func(y_preds, y)
        losses.update(loss.item(), batch_size)
        y_preds = softmax(y_preds)
        preds = np.vstack((preds, y_preds.to('cpu').numpy()))
        end = time.time()

    return losses.avg, preds

# %% [markdown]
# ## Train Loop

# %%
def train_loop(cfg):
    train_df = pd.read_csv(paths.DATASET + 'train_df.csv')
    train_df['total_evaluators'] = train_df[['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']].sum(axis=1)

    kaggle_spectrograms = np.load(paths.DATASET + 'kaggle_specs.npy', allow_pickle=True).item()
    eeg_spectrograms = np.load(paths.DATASET + 'eeg_specs.npy', allow_pickle=True).item()

    train_folds, valid_folds = train_test_split(train_df, test_size=0.2, random_state=42)
    train_dataset = EEGDataset(train_folds, kaggle_spectrograms=kaggle_spectrograms, eeg_spectrograms=eeg_spectrograms, mode="train", augment=cfg['augment'])
    valid_dataset = EEGDataset(valid_folds, kaggle_spectrograms=kaggle_spectrograms, eeg_spectrograms=eeg_spectrograms, mode="train", augment='none')
    
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['batch_size'],
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg['batch_size'],
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)
    
    model = EEGNet(cfg)
    model.to(device)

    if cfg['optimizer'] == 'adam':
        optimizer = AdamW(model.parameters(), lr=0.1, weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'adan':
        optimizer = Adan(model.parameters(), lr=0.1, weight_decay=cfg['weight_decay'])

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg['max_lr'],
        epochs=cfg['epochs'],
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )

    # Load existing checkpoint through `get_checkpoint()` API.
    if train.get_checkpoint():
        loaded_checkpoint = train.get_checkpoint()
        with loaded_checkpoint.as_directory() as loaded_checkpoint_dir:
            model_state, optimizer_state = torch.load(
                os.path.join(loaded_checkpoint_dir, "checkpoint.pt")
            )
            model.load_state_dict(model_state)
            optimizer.load_state_dict(optimizer_state)
    
    best_preds = None
    best_loss = np.inf
    
    train_losses = []
    val_losses = []
    
    for epoch in range(cfg['epochs']):
        avg_train_loss = train_epoch(train_loader, model, optimizer, epoch, scheduler)
        avg_val_loss, preds = valid_epoch(valid_loader, model)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": avg_val_loss,
                "train_loss": avg_train_loss},
                checkpoint=checkpoint,
            )


# %% [markdown]
# # Losses

# %%
def main(num_samples=100, max_num_epochs=10, gpus_per_trial=1):
  config = {
    'model_name': tune.choice(['efficientnet_b0', 'efficientnet_b2', 'efficientnet_b3']),
    'drop_rate': tune.uniform(.0, .9),
    'drop_path_rate': tune.uniform(.0, .9),
    'max_lr': tune.loguniform(1e-4, 0.1),
    'weight_decay': tune.loguniform(1e-3, 0.1),
    'batch_size': tune.choice([4,8,16,32,64]),
    'epochs': tune.choice([6, 8, 10, 12, 15, 20]),
    'channels': tune.choice([1,3]),
    'optimizer': tune.choice(['adam', 'adan']),
    'augment': tune.choice(['none', 'flip', 'mask', 'both'])
  }
  algo = OptunaSearch()
  algo = ConcurrencyLimiter(algo, max_concurrent=2)

  tuner = tune.Tuner(
    tune.with_resources(
      tune.with_parameters(train_loop),
      resources={"cpu": 2, "gpu": gpus_per_trial}
    ),
    tune_config=tune.TuneConfig(
      metric="loss",
      mode="min",
      search_alg=algo,
      num_samples=num_samples
    ),
    run_config=train.RunConfig(
      storage_path=Path("/home/jdusseljee/efficientnet/results").resolve(), 
      name="experiment4",
      log_to_file="/home/jdusseljee/efficientnet/experiment4.log"
      ),
    param_space=config,
  )
  results = tuner.fit()

  best_result = results.get_best_result("loss", "min")
  print("Best trial config: {}".format(best_result.config))
  print("Best trial final validation loss: {}".format(
      best_result.metrics["loss"]))


main()


