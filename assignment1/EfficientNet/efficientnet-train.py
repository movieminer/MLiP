# %%
import albumentations as A
import gc
import matplotlib.pyplot as plt
import math
import numpy as np
import os
import pandas as pd
import random
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import GroupShuffleSplit, GroupKFold

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

os.environ["CUDA_VISIBLE_DEVICES"] = "3"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

# %% [markdown]
# # Config

# %%
class config:
    DEVICE = 'cuda'
    AMP = True
    PRINT_FREQ = 20
    NUM_WORKERS = 0 # multiprocessing.cpu_count()
    MAX_GRAD_NORM = 1e7

    MODEL = "efficientnet_b3"
    FREEZE = False
    NUM_FROZEN_LAYERS = 39
    
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VALID = 32
    EPOCHS = 12
    FOLDS = 5
    VAL_SIZE = .2   # If FOLDS=1 
    WEIGHT_DECAY = 0.016329

    SEED = 20
    TRAIN_FULL_DATA = False
    VISUALIZE = True
    
    
class paths:
    OUTPUT_DIR = "./working2/"
    DATASET = '../input/dataset/'

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
# # Load data

# %%
train_df = pd.read_csv(paths.DATASET + 'train_df.csv')

# %%
kaggle_spectrograms = np.load(paths.DATASET + 'kaggle_specs.npy', allow_pickle=True).item()
eeg_spectrograms = np.load(paths.DATASET + 'eeg_specs.npy', allow_pickle=True).item()

# %% [markdown]
# ## Create folds

# %%
if config.FOLDS > 1:
    gkf = GroupKFold(n_splits=config.FOLDS)
    for fold, (train_index, valid_index) in enumerate(gkf.split(train_df, train_df.target, train_df.patient_id)):
        train_df.loc[valid_index, "fold"] = int(fold)

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
    def __init__(self):
        super(ReshapeInput, self).__init__()

    def forward(self, x):
        # Split the 8-channel input into individual channels
        channels = [x[:, :, :, i:i+1] for i in range(8)]
        
        # Concatenate the first 4 channels along the height dimension
        x1 = torch.cat(channels[:4], dim=1)
        x2 = torch.cat(channels[4:], dim=1)
        x3 = torch.cat([x1,x2], dim=2)
        
        # Concatenate the channels along the width dimension to form a 3-channel 512x512 image
        x = torch.cat([x3], dim=3)  # Stack the same image thrice along the channel dimension
        x = x.permute(0, 3, 1, 2)

        return x

class EEGNet(nn.Module):
    def __init__(
            self,
            model_name: str,
            pretrained: bool,
        ):
        super().__init__()
        self.reshape = ReshapeInput()
        self.model = timm.create_model(
            model_name=model_name, pretrained=pretrained,
            num_classes=6, in_chans=1, drop_rate=.183, drop_path_rate=.524)

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

# %% [markdown]
# ## Train and Val functions

# %%
def train_epoch(train_loader, model, optimizer, epoch, scheduler):
    device = config.DEVICE
    model.train() 
    loss_func = KLDivLossWithLogits(reduction="batchmean")
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
    losses = AverageMeter()
    
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, (X, y) in enumerate(tqdm_train_loader):
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

    print(f'Train loss: {losses.avg}')
    return losses.avg


def valid_epoch(valid_loader, model):
    device = config.DEVICE
    model.eval()
    loss_func = KLDivLossWithLogits(reduction="batchmean")
    softmax = nn.Softmax(dim=1)
    losses = AverageMeter()
    preds = np.empty((0,6))
    start = end = time.time()
    with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
        for step, (X, y) in enumerate(tqdm_valid_loader):
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

    print(f'Val loss: {losses.avg}')
    return losses.avg, preds

# %% [markdown]
# ## Train Loop

# %%
def train_loop(df, fold):
    if config.FOLDS > 1:
        train_folds = df[df['fold'] != fold].reset_index(drop=True)
        valid_folds = df[df['fold'] == fold].reset_index(drop=True)

        train_dataset = EEGDataset(train_folds, kaggle_spectrograms=kaggle_spectrograms, eeg_spectrograms=eeg_spectrograms, mode="train", augment="both")
        valid_dataset = EEGDataset(valid_folds, kaggle_spectrograms=kaggle_spectrograms, eeg_spectrograms=eeg_spectrograms, mode="train", augment="none")
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=config.VAL_SIZE, random_state=config.SEED)
        train_index, valid_index = next(gss.split(df, df.target, groups=df['patient_id']))

        train_entries = df.iloc[train_index].reset_index(drop=True)
        valid_entries = df.iloc[valid_index].reset_index(drop=True)

        train_dataset = EEGDataset(train_entries, kaggle_spectrograms=kaggle_spectrograms, eeg_spectrograms=eeg_spectrograms, mode="train", augment="both")
        valid_dataset = EEGDataset(valid_entries, kaggle_spectrograms=kaggle_spectrograms, eeg_spectrograms=eeg_spectrograms, mode="train", augment="none")
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_VALID,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)
    
    model = EEGNet(model_name=config.MODEL, pretrained=True)
    model.to(device)
    # model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=config.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=0.0067194,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )
    
    best_preds = None
    
    best_loss = np.inf
    
    train_losses = []
    val_losses = []
    
    for epoch in range(config.EPOCHS):
        avg_train_loss = train_epoch(train_loader, model, optimizer, epoch, scheduler)
        avg_val_loss, preds = valid_epoch(valid_loader, model)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        
        if avg_val_loss < best_loss:
            best_preds = preds
            best_loss = avg_val_loss
            torch.save(model.state_dict(), paths.OUTPUT_DIR + f"/best_model_fold{fold}.pth")

    valid_folds[target_preds] = best_preds

    torch.cuda.empty_cache()
    gc.collect()
    
    return valid_folds, train_losses, val_losses

# %% [markdown]
# # Train

# %%

def get_result(oof_df):
    kl_loss = KLDivLossWithLogits(reduction="batchmean")
    labels = torch.tensor(oof_df[label_cols].values)
    preds = torch.tensor(oof_df[target_preds].values)
    result = kl_loss(preds, labels)
    return result

oof_df = pd.DataFrame()
train_losses = []
val_losses = []
for fold in range(config.FOLDS):
    if fold in [0, 1, 2, 3, 4]:
        _oof_df, train_losses_fold, val_losses_fold = train_loop(train_df, fold)
        train_losses.append(train_losses_fold)
        val_losses.append(val_losses_fold)
        oof_df = pd.concat([oof_df, _oof_df])
        print(f"========== Fold {fold} result: {get_result(_oof_df)} ==========")
oof_df = oof_df.reset_index(drop=True)
LOGGER.info(f"========== CV: {get_result(oof_df)} ==========")
oof_df.to_csv(paths.OUTPUT_DIR + '/oof_df.csv', index=False)

# %% [markdown]
# ## CV Score

# %%
import sys
sys.path.append('/kaggle/input/kaggle-kl-div')
from kaggle_kl_div import score

oof = oof_df.copy()
for target in ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']:
    oof[target] = oof[target + '_pred']
oof = oof.loc[:,['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
oof['id'] = np.arange(len(oof))

true = oof_df.copy()
true = true.loc[:,['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']]
true['id'] = np.arange(len(true))

cv = score(solution=true, submission=oof, row_id_column_name='id')
print('CV Score KL-Div for EfficientNetB2 =',cv)

# %% [markdown]
# # Losses

# %%
avg_train_losses = np.mean(np.array(train_losses), 0)
avg_val_losses = np.mean(np.array(val_losses), 0)

plt.plot(avg_train_losses, 'b', avg_val_losses, 'r')


