# %%
import pandas as pd
import numpy as np
import os
import torch
import random
import gc
import math
import torch.nn as nn
from torch.optim.optimizer import Optimizer
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
import time
import scipy.io as io

from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from scipy import signal

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
os.environ["TORCH_USE_CUDA_DSA"] = "1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

# %% [markdown]
# # Config

# %%
class config:
    DEVICE = 'cuda'
    AMP = True
    NUM_WORKERS = 4 # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    TRAIN_FULL_DATA = False
    VISUALIZE = True
    MAX_GRAD_NORM = 1e7

    FOLDS = 5
    VAL_SIZE = .2   # If FOLDS=1 
    BATCH_SIZE_TRAIN = 32
    BATCH_SIZE_VALID = 32
    EPOCHS = 30
    MAX_LR = 0.01
    DROPOUT=0.6
    FC=100
    RATIO=16
    WEIGHT_DECAY=0.01
    PCT_START=0.10
    FINAL_DIV_FACTOR=100
    
class paths:
    OUTPUT_DIR = "/kaggle/working/"
    TRAIN_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/train.csv"
    PRELOADED_EEGS = "/kaggle/input/brain-eegs/train_eegs.npy"
    SCALING_FILTER = "/kaggle/input/daubechies-4-scaling-filters/"

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
# Loading train csv
train_csv = pd.read_csv(paths.TRAIN_CSV)

# Loading EEGs
train_eegs = np.load(paths.PRELOADED_EEGS, allow_pickle=True).item()

print("Data successfully loaded")

# %%
# Create modified train csv
train_csv['total_evaluators'] = train_csv[label_cols].sum(axis=1)

train_df = train_csv.groupby('eeg_id')[['total_evaluators']].agg({
    'total_evaluators': 'mean'
})
train_df.columns = ['total_evaluators']

aux = train_csv.groupby('eeg_id')[['patient_id']].agg('first')
train_df['patient_id'] = aux

aux = train_csv.groupby('eeg_id')[label_cols].agg('sum')
for label in label_cols:
    train_df[label] = aux[label].values
    
y_data = train_df[label_cols].values
y_data = y_data / y_data.sum(axis=1,keepdims=True)
train_df[label_cols] = y_data

aux = train_csv.groupby('eeg_id')[['expert_consensus']].agg('first')
train_df['target'] = aux

train_df = train_df.reset_index()


# %% [markdown]
# ## Create folds

# %%
if config.FOLDS > 1:
    gkf = GroupKFold(n_splits=config.FOLDS)
    for fold, (train_index, valid_index) in enumerate(gkf.split(train_df, train_df.target, train_df.patient_id)):
        train_df.loc[valid_index, "fold"] = int(fold)

# %% [markdown]
# # Dataset
# 

# %%
class EEGDataset(Dataset):
    def __init__(self, train_df, eegs, mode='train', augment=True):
        self.train_df = train_df
        self.eegs = eegs
        self.mode = mode
        self.augment = augment

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
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
        
        if self.mode != 'test':
            y = torch.from_numpy(row[label_cols].values.astype(np.float32))

        return torch.from_numpy(X), y


# %% [markdown]
# # Model

# %%
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
        
        result = self.dropout(self.elu(self.fc1(flat)))
        result = self.fc2(result)
        
        return result

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
# ## Optimizer (Adan)

# %%
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
    
#     with tqdm(train_loader, unit="train_batch", desc='Train') as tqdm_train_loader:
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
#     with tqdm(valid_loader, unit="valid_batch", desc='Validation') as tqdm_valid_loader:
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

    print(f'Val loss: {losses.avg}')
    return losses.avg, preds

# %% [markdown]
# ## Train Loop

# %%
def train_loop(df, fold):    
    if config.FOLDS > 1:
        train_folds = df[df['fold'] != fold].reset_index(drop=True)
        valid_folds = df[df['fold'] == fold].reset_index(drop=True)

        train_dataset = EEGDataset(train_folds, eegs=train_eegs, mode="train", augment=True)
        valid_dataset = EEGDataset(valid_folds, eegs=train_eegs, mode="train", augment=False)
    else:
        gss = GroupShuffleSplit(n_splits=1, test_size=config.VAL_SIZE, random_state=config.SEED)
        train_index, valid_index = next(gss.split(df, df.target, groups=df['patient_id']))

        train_entries = df.iloc[train_index].reset_index(drop=True)
        valid_entries = df.iloc[valid_index].reset_index(drop=True)

        train_dataset = EEGDataset(train_entries, eegs=train_eegs, mode="train", augment=True)
        valid_dataset = EEGDataset(valid_entries, eegs=train_eegs, mode="train", augment=False)
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE_TRAIN,
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=config.BATCH_SIZE_VALID,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)
    
   
    model = EEG3DCBAMNet()
    model.to(device)
    model = nn.DataParallel(model)

    optimizer = Adan(model.parameters(), lr=config.MAX_LR, weight_decay=config.WEIGHT_DECAY)
    
    scheduler = OneCycleLR(
        optimizer,
        max_lr=config.MAX_LR,
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
        print(f"EPOCH {epoch}")
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
        print(f"========================= Fold {fold} =========================")
        _oof_df, train_losses_fold, val_losses_fold = train_loop(train_df, fold)
        train_losses.append(train_losses_fold)
        val_losses.append(val_losses_fold)
        oof_df = pd.concat([oof_df, _oof_df])
        print(f"========== Fold {fold} result: {get_result(_oof_df)} ==========")
oof_df = oof_df.reset_index(drop=True)
LOGGER.info(f"========== CV: {get_result(oof_df)} ==========")
oof_df.to_csv(paths.OUTPUT_DIR + '/oof_df.csv', index=False)

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
print('CV Score KL-Div for EEG3DNet =',cv)


