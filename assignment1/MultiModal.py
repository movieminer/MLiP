# %%
import albumentations as A
import gc
import matplotlib.pyplot as plt
import math
import multiprocessing
import numpy as np
import os
import scipy.io as io
import pandas as pd
import random
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR, StepLR
from sklearn.model_selection import train_test_split, KFold, GroupKFold

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from typing import Dict, List
from scipy import signal
from torchsummary import summary

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

# %% [markdown]
# # Config

# %%
class config:
    DEVICE = 'cuda'
    AMP = True
    BATCH_SIZE_TRAIN = 256
    BATCH_SIZE_VALID = 256
    EPOCHS = 15
    FOLDS = 5
    VAL_SIZE = .2   # If FOLDS=1 
    FREEZE = False
    MAX_GRAD_NORM = 1e7
    MODEL = "efficientnet_b3"
    NUM_FROZEN_LAYERS = 39
    NUM_WORKERS = 0 # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    TRAIN_FULL_DATA = False
    VISUALIZE = True
    WEIGHT_DECAY = 0.016329
    RATIO = 32
    FC = 66
    DROPOUT = 0.47170
    
    
class paths:
    OUTPUT_DIR = "/vol/tensusers3/jdusseljee/efficientnet/train_HPO/multimodal/"
    DATASET = '/vol/tensusers/jdusseljee/efficientnet/input/dataset/'
    EEGS_NPY = '/vol/tensusers/thijsdejong/EEG3DNet/input/dataset/train_eegs.npy'
    SCALING_FILTER = '/vol/tensusers3/jdusseljee/efficientnet/train_HPO/scaling_filter.mat'
    EEGNET_WEIGHTS = '/vol/tensusers3/jdusseljee/efficientnet/train_HPO/pop2_same_lr/'
    CBAMNET_WEIGHTS = '/vol/tensusers/thijsdejong/EEG3DNet/working/'

if not os.path.exists(paths.OUTPUT_DIR):
    os.makedirs(paths.OUTPUT_DIR)
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
eegs = np.load(paths.EEGS_NPY, allow_pickle=True).item()

# %% [markdown]
# ## Create folds

# %%
if config.FOLDS > 1:
    gkf = GroupKFold(n_splits=config.FOLDS)
    for fold, (train_index, valid_index) in enumerate(gkf.split(train_df, train_df.target, train_df.patient_id)):
        train_df.loc[valid_index, "fold"] = int(fold)

# %% [markdown]
#mDataset

# %%
class EEGDataset(Dataset):
    def __init__(self, train_df, kaggle_spectrograms, eeg_spectrograms, eegs, mode='train', augment=True):
        self.train_df = train_df
        self.kaggle_specs = kaggle_spectrograms
        self.eeg_specs = eeg_spectrograms
        self.eegs = eegs
        self.mode = mode
        self.augment = augment

    def __len__(self):
        return len(self.train_df)
    
    def __get_y(self, idx):
        row = self.train_df.iloc[idx]
        y = row[label_cols].values.astype(np.float32)

        return torch.from_numpy(y)

    def __get_spectrograms(self, idx):
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

        return torch.from_numpy(X)
    
    def __get_raw(self, idx):
        #idx = idx%(config.BATCH_SIZE_TRAIN if self.mode == 'train' else config.BATCH_SIZE_VALID)
        row = self.train_df.iloc[idx]

        eeg = self.eegs[row['eeg_id']]
        eeg_length = len(eeg)
        
        X = np.zeros((19,1,6400),dtype='float32')

        ORDER = ['Fp1','Fp2','F7', 'F3', 'Fz', 'F4', 'F8','T3', 'C3', 'Cz', 'C4', 'T4','T5', 'P3', 'Pz', 'P4', 'T6','O1','O2']      
        
        if self.augment and self.mode == 'train':
            # Randomly flip order
            if np.random.rand() > 0.5:
                ORDER = ['Fp1','Fp2','F8', 'F4', 'Fz', 'F3', 'F7','T4', 'C4', 'Cz', 'C3', 'T3','T6', 'P4', 'Pz', 'P3', 'T5','O1','O2']      

            time_shift = random.randint(-int(eeg_length * 0.1), int(eeg_length * 0.1))  # Shift by 10% of the length
            
        
        for i in range(19):
            
            val = eeg[ORDER[i]].values.astype('float32')
                       
            m = np.nanmean(val)

            if np.isnan(val).mean()<1: 
                val = np.nan_to_num(val,nan=m)
            else: 
                val[:] = 0
               
            val = signal.resample(val, 6400)
            
            val = (val-np.mean(val))/(np.std(val)+1e-6)
            
            # Augment
            if self.augment and self.mode == 'train':
                if time_shift > 0:
                    val = np.roll(val, time_shift, axis=0)  # Shift right
                elif time_shift < 0:
                    val = np.roll(val, -time_shift, axis=0)  # Shift left
                    # Add Gaussian noise if augmenting
                noise = np.random.normal(0, 0.01, size=val.shape)  # Mean = 0, Std = 0.01
                val += noise
            
            X[i,0,:] = val

        return torch.from_numpy(X)

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

        if self.augment:
            transforms = A.Compose([
                A.XYMasking(**params1, p=0.3),
                A.XYMasking(**params2, p=0.3),
                A.XYMasking(**params3, p=0.3),
                A.HorizontalFlip(p=0.5),
            ])
            return transforms(image=img)['image']
        else:
            return img
    
    def __getitem__(self, idx):
        X_raw = self.__get_raw(idx)
        X_spec = self.__get_spectrograms(idx)
        X = (X_raw, X_spec)

        if self.mode == 'train':
            y = self.__get_y(idx)

            return X, y
        else:
            return X


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
    

# Model
# # MultiLevel Spectral, splitting the signal into different frequency bands
class MultiLevel_Spectral(nn.Module): 
    def __init__(self, inc, params_path=f'{paths.SCALING_FILTER}'):
        super(MultiLevel_Spectral, self).__init__()          
        
        self.filter_length = io.loadmat(params_path)['Lo_D'].shape[1]
        self.conv = nn.Conv2d(in_channels = inc, 
                              out_channels = inc*2, 
                              kernel_size = (1, self.filter_length), 
                              stride = (1, 2), padding = 0, 
                              groups = inc,
                              bias = False)
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.ConvTranspose2d):
                f = io.loadmat(params_path)
                Lo_D, Hi_D = np.flip(f['Lo_D'], axis = 1).astype('float32'), np.flip(f['Hi_D'], axis = 1).astype('float32')
                m.weight.data = torch.from_numpy(np.concatenate((Lo_D, Hi_D), axis = 0)).unsqueeze(1).unsqueeze(1).repeat(inc, 1, 1, 1)            
                m.weight.requires_grad = False 
    
    def self_padding(self, x):
        return torch.cat((x[:, :, :, -(self.filter_length//2-1):], x, x[:, :, :, 0:(self.filter_length//2-1)]), (self.filter_length//2-1))
                           
    def forward(self, x):
        out = self.conv(self.self_padding(x))
        return out[:, 0::2,:, :], out[:, 1::2, :, :]


# # Convolutional Block Attention Modules (Channel, Spatial, BasicBlock)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=config.RATIO):
        super(ChannelAttention, self).__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.max_pool = nn.AdaptiveMaxPool3d(1)
           
        self.fc = nn.Sequential(nn.Conv3d(in_planes, in_planes // ratio, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv3d(in_planes // ratio, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv3d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)

    
class BasicBlock(nn.Module):
    def __init__(self, kernel_size, inplanes, planes, stride=1):
        super(BasicBlock, self).__init__()
        
        self.kernel_size = kernel_size
        self.inplanes = inplanes
        self.planes = planes
        self.stride = stride
        
        self.conv = nn.Conv3d(self.inplanes, self.planes, self.kernel_size, 
                              stride=self.stride, padding="same")
        self.bn = nn.BatchNorm3d(self.planes)

        self.ca = ChannelAttention(self.planes)
        self.sa = SpatialAttention()

        self.maxpool = nn.MaxPool3d(kernel_size=self.kernel_size, stride=(1,1,2))
        
        self.elu = nn.ELU(inplace=True)
        

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.elu(out)
        
        out = self.ca(out) * out
        out = self.sa(out) * out
        
        out = self.maxpool(out)
        
        return out


# # MapSpatial, mapping the spatial features into a 3D tensor
class MapSpatial(nn.Module):
    def __init__(self):
        super(MapSpatial, self).__init__()
        
    def forward(self, x):
        m = torch.mean(x,1,True)
        c = torch.split(x,1,dim=1)

        l1 = torch.cat((m,c[0],m,c[1],m),1)
        l2 = torch.cat(c[2:7],1)
        l3 = torch.cat(c[7:12],1)
        l4 = torch.cat(c[12:17],1)
        l5 = torch.cat((m,c[17],m,c[18],m),1)
        
        spat = torch.cat([l1,l2,l3,l4,l5],2)
        spat = spat[:,None]

        return spat
        

# # ParallelModel, one for each frequency band
class ParallelModel(nn.Module):
    def __init__(self, kernel_size):
        super(ParallelModel, self).__init__()
        self.kernel_size = kernel_size
        
        self.layer1 = BasicBlock(self.kernel_size,1,32)
        self.layer2 = BasicBlock(self.kernel_size,32,64)
        self.layer3 = BasicBlock(self.kernel_size,64,128)
 
        self.conv = nn.Conv3d(128, 128, kernel_size=self.kernel_size, 
                              stride=1, padding="same")
    
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.elu(self.conv(out))
        return out
    

# # EEG3DCBAMNet, the main model   
class EEG3DCBAMNet(nn.Module):    
    def __init__(self):
        super().__init__()
        
        self.spectral = MultiLevel_Spectral(19)
        
        self.mapspatial = MapSpatial()
       
        self.dwt = MultiLevel_Spectral(19)
    
        self.gammaband = ParallelModel((2,2,7))
        self.betaband = ParallelModel((2,2,7))
        self.alphaband = ParallelModel((2,2,7))
        self.thetaband = ParallelModel((2,2,3))
        self.deltaband = ParallelModel((2,2,3))
                
        self.flatten = nn.Flatten(start_dim=1)
        
        self.reshape = nn.AdaptiveAvgPool3d(1)
        
        self.fc1 = nn.Linear(640,config.FC)
        self.fc2 = nn.Linear(config.FC,6)
        
        self.dropout = nn.Dropout(config.DROPOUT)
        
        self.elu = nn.ELU(inplace=True)
        
    def forward(self, x):
        out, gamma = self.dwt(x)
        out, beta = self.dwt(out)
        out, alpha = self.dwt(out)
        delta, theta = self.dwt(out)
        
        gamma = self.mapspatial(gamma)
        beta = self.mapspatial(beta)
        alpha = self.mapspatial(alpha)
        theta = self.mapspatial(theta)
        delta = self.mapspatial(delta)

        x1 = self.gammaband(gamma)
        x2 = self.betaband(beta)
        x3 = self.alphaband(alpha)
        x4 = self.thetaband(theta)
        x5 = self.deltaband(delta)
        
        x1, x2, x3, x4, x5 = self.reshape(x1), self.reshape(x2), self.reshape(x3), self.reshape(x4), self.reshape(x5)
        
        cat_f = torch.cat((x1, x2,x3,x4,x5), 1)
        
        flat = self.flatten(cat_f)
        
        # result = self.dropout(self.elu(self.fc1(flat)))
        # result = self.fc2(result)
        
        return flat


class MultiModalNet(nn.Module):
    def __init__(self, eegnet_weights_path=None, cbamnet_weights_path=None, feature_dim=256, freeze=False):
        super().__init__()
        eegnet_full = EEGNet(model_name=config.MODEL, pretrained=False)
        cbamnet_full = EEG3DCBAMNet()

        eegnet_feature_size = 262144
        cbamnet_feature_size = 640

        # Load model weights
        if eegnet_weights_path:
            eegnet_full.load_state_dict(torch.load(eegnet_weights_path))
        if cbamnet_weights_path:
            cbamnet_full.load_state_dict(torch.load(cbamnet_weights_path))

        # Remove last layer from both models
        *eegnet_layers, _ = list(eegnet_full.children())
        self.eegnet = nn.Sequential(*eegnet_layers)

        # *cbamnet_layers, _, _ = list(cbamnet_full.children())
        # self.cbamnet = nn.Sequential(*cbamnet_layers)
        self.cbamnet = cbamnet_full

        # Freeze layers if needed
        if freeze:
            for param in self.eegnet.parameters():
                param.requires_grad = False
            for param in self.cbamnet.parameters():
                param.requires_grad = False

        self.flatten = nn.Flatten(start_dim=1)
        self.dropout = nn.Dropout(p=0.6)

        self.fc1 = nn.Linear(eegnet_feature_size + cbamnet_feature_size, feature_dim)
        self.fc2 = nn.Linear(feature_dim, 64)
        self.fc3 = nn.Linear(64, 6)

    def forward(self, X):
        
        spec_features = self.eegnet(X[1])
        spec_features = self.flatten(spec_features)
        cbam_features = self.cbamnet(X[0])

        features = torch.cat((spec_features, cbam_features), dim=1)

        x = F.relu(self.fc1(features))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)

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
    global_step = 0
    
    with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
        for step, (X, y) in enumerate(tqdm_train_loader):
            X = (X[0].to(device), X[1].to(device))
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
            X = (X[0].to(device), X[1].to(device))
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
    train_folds = df[df['fold'] != fold].reset_index(drop=True)
    valid_folds = df[df['fold'] == fold].reset_index(drop=True)

    train_dataset1 = EEGDataset(train_folds[train_folds['total_evaluators'] < 10], kaggle_spectrograms=kaggle_spectrograms, eeg_spectrograms=eeg_spectrograms, eegs=eegs, mode="train", augment="both")
    train_dataset2 = EEGDataset(train_folds[train_folds['total_evaluators'] >= 10], kaggle_spectrograms=kaggle_spectrograms, eeg_spectrograms=eeg_spectrograms, eegs=eegs, mode="train", augment="both")
    valid_dataset = EEGDataset(valid_folds, kaggle_spectrograms=kaggle_spectrograms, eeg_spectrograms=eeg_spectrograms, eegs=eegs, mode="train", augment="none")
    
    train_loader1 = DataLoader(train_dataset1,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    train_loader2 = DataLoader(train_dataset2,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_VALID,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)
    
    eegnet_weights = f'{paths.EEGNET_WEIGHTS}best_model_fold{fold}.pth'
    cbamnet_weights = f'{paths.CBAMNET_WEIGHTS}best_model_fold{fold}.pth'
    model = MultiModalNet(eegnet_weights_path=eegnet_weights, cbamnet_weights_path=cbamnet_weights, freeze=True)
    model.to(device)
    # model = nn.DataParallel(model)


    valid_folds1, train_losses1, val_losses1 = train_stage(train_loader1, valid_loader, valid_folds, model, 0.005)
    valid_folds2, train_losses2, val_losses2 = train_stage(train_loader2, valid_loader, valid_folds, model, 0.005)

    valid_folds = valid_folds1 + valid_folds2
    train_losses = train_losses1 + train_losses2
    val_losses = val_losses1 + val_losses2

    return valid_folds, train_losses, val_losses


def train_stage(train_loader, valid_loader, valid_folds, model, max_lr):
    optimizer = torch.optim.AdamW(model.parameters(), lr=0.01, weight_decay=config.WEIGHT_DECAY)
    scheduler = OneCycleLR(
        optimizer,
        max_lr=max_lr,
        epochs=config.EPOCHS,
        steps_per_epoch=len(train_loader),
        pct_start=0.1,
        anneal_strategy="cos",
        final_div_factor=100,
    )
    # scheduler = StepLR(
    #     optimizer,
    #     step_size=5
    # )
    
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
            torch.save(model.state_dict(), paths.OUTPUT_DIR + f"best_model_fold{fold}.pth")

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


