# %%
import gc
import numpy as np
import os
import pandas as pd
import random
import torch
import torch.nn as nn
import scipy.io as io


from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from scipy import signal


os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print('Using', torch.cuda.device_count(), 'GPU(s)')

# %% [markdown]
# # Config

# %%
class config:
    BATCH_SIZE = 32
    NUM_WORKERS = 4 # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = False
    
    
class paths:
    OUTPUT_DIR = "/kaggle/working/"
    TEST_CSV = "/kaggle/input/hms-harmful-brain-activity-classification/test.csv"
    TEST_EEGS= "/kaggle/input/hms-harmful-brain-activity-classification/test_eegs/"
    SCALING_FILTER = "/kaggle/input/daubechies-4-scaling-filters/"
    
model_weights = [x for x in glob("/kaggle/input/baseline-3d-eeg/best_model_fold*.pth")]
model_weights

# %% [markdown]
# # Util

# %%
def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed) 

    
label_to_num = {'Seizure': 0, 'LPD': 1, 'GPD': 2, 'LRDA': 3, 'GRDA': 4, 'Other':5}
num_to_label = {v: k for k, v in label_to_num.items()}
seed_everything(config.SEED)

# %% [markdown]
# # Load data

# %%
test_df = pd.read_csv(paths.TEST_CSV)
print(f"Test dataframe shape is: {test_df.shape}")
test_df.head()

# %% [markdown]
# # Dataset

# %%
class EEGDatasetSplit(Dataset):
    def __init__(self, train_df, mode='train', augment=True):
        self.train_df = train_df
        self.mode = mode
        self.augment = augment

    def __len__(self):
        return len(self.train_df)

    def __getitem__(self, idx):
        
        row = self.train_df.iloc[idx]
        eeg_id = row['eeg_id']
        eeg = pd.read_parquet(f'{paths.TEST_EEGS}{eeg_id}.parquet')
        X = np.zeros((19,1,6400),dtype='float32')

        ORDER = ['Fp1','Fp2','F7', 'F3', 'Fz', 'F4', 'F8','T3', 'C3', 'Cz', 'C4', 'T4','T5', 'P3', 'Pz', 'P4', 'T6','O1','O2']                
        
        for i in range(19):
            val = eeg[ORDER[i]].values.astype('float32')
                       
            m = np.nanmean(val)
            if np.isnan(val).mean()<1: 
                val = np.nan_to_num(val,nan=m)
            else: 
                val[:] = 0
               
            val = signal.resample(val, 6400)
            val = (val-np.mean(val))/(np.std(val)+1e-6)
            
            X[i,0,:] = val
        
        del eeg
        
        return torch.from_numpy(X)

# %% [markdown]
# # Model

# %%
class MultiLevel_Spectral(nn.Module): 
    def __init__(self, inc, params_path=f'{paths.SCALING_FILTER}scaling_filter.mat'):
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

# Convolutional Block Attention Modules (Channel, Spatial, BasicBlock)
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
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
        
        self.fc1 = nn.Linear(640,100)
        self.fc2 = nn.Linear(100,6)
        
        self.dropout = nn.Dropout(0.6)
        
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
# # Inference function

# %%
def inference_function(test_loader, model, device):
    model.eval()
    softmax = nn.Softmax(dim=1)
    prediction_dict = {}
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, X in enumerate(tqdm_test_loader):
            X = X.to(device)
            with torch.no_grad():
                y_preds = model(X)
            y_preds = softmax(y_preds)
            preds.append(y_preds.to('cpu').numpy()) 
                
    prediction_dict["predictions"] = np.concatenate(preds) 
    return prediction_dict

# %% [markdown]
# # Infer

# %%
predictions = []

for model_weight in model_weights:
    test_dataset = EEGDatasetSplit(test_df, mode="test")
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True, drop_last=False
    )
    model = EEG3DCBAMNet()
    model.to(device)
    
    checkpoint = torch.load(model_weight)
    model.load_state_dict(checkpoint)
    prediction_dict = inference_function(test_loader, model, device)
    predictions.append(prediction_dict["predictions"])
    torch.cuda.empty_cache()
    gc.collect()
    
predictions = np.array(predictions)
predictions = np.mean(predictions, axis=0)

# %% [markdown]
# # <b><span style='color:#F1A424'>|</span> Save Submission</b><a class='anchor' id='submission'></a> [↑](#top) 
# 
# ***

# %%
TARGETS = ['seizure_vote', 'lpd_vote', 'gpd_vote', 'lrda_vote', 'grda_vote', 'other_vote']
sub = pd.DataFrame({'eeg_id': test_df.eeg_id.values})
sub[TARGETS] = predictions
sub.to_csv('submission.csv',index=False)
print(f'Submission shape: {sub.shape}')
sub.head()


