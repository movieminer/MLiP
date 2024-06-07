# %%
import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
os.environ['TUNE_RESULT_DIR'] = '/vol/tensusers3/thijsdejong/planttraits/results'


from PIL import Image
from numpy import asarray
import albumentations as A
import math
import numpy as np
import pandas as pd
import random
import time
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR
from sklearn.model_selection import train_test_split, GroupKFold
from sklearn.preprocessing import StandardScaler
import torchmetrics
import cv2
import tempfile
from pathlib import Path


from albumentations.pytorch import ToTensorV2
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from ray import train, tune
from ray.train import Checkpoint
from ray.tune.search import ConcurrencyLimiter
from ray.tune.search.optuna import OptunaSearch

device = torch.device("cuda:0")
NAME = "experiment-1"
print(device)
print('Using', torch.cuda.device_count(), 'GPU(s)')

# %% [markdown]
# # Config

# %%
class config:
    DEVICE = 'cuda'
    AMP = True
    FOLDS = 1
    VAL_SIZE = .2   # If FOLDS=1 
    FREEZE = False
    MAX_GRAD_NORM = 1e7
    NUM_WORKERS = 0 # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    VISUALIZE = True
    IMG_SIZE = 384
    VIT_TRAINED = False
    
    
class paths:
    OUTPUT_DIR = '/vol/tensusers3/thijsdejong/planttraits/output'
    DATASET = '/vol/tensusers3/thijsdejong/planttraits/data/'
    IMAGES = '/vol/tensusers3/thijsdejong/planttraits/data/train_images/'
    IMAGES_TEST = '/vol/tensusers3/thijsdejong/planttraits/data/test_images/'
    PONYHOME = '/vol/tensusers3/thijsdejong'
    
tempfile.tempdir = f'{paths.PONYHOME}/planttraits/tmp'

# %% [markdown]
# # Utils

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
    
LOGGER = get_logger()
seed_everything(config.SEED)

# %%
target_cols = ['X4_mean', 'X11_mean', 'X18_mean', 'X26_mean', 'X50_mean', 'X3112_mean']
target_preds = [x + "_pred" for x in target_cols]
aux_class_names = list(map(lambda x: x.replace("mean","sd"), target_cols))

# %% [markdown]
# # Data

# %%
df = pd.read_csv(paths.DATASET + 'train.csv')
df['image_files'] = df['id'].apply(lambda x: paths.IMAGES +str(x)+'.jpeg')
df.loc[:, aux_class_names] = df.loc[:, aux_class_names].fillna(-1)
FEATURE_COLS = df.columns[1:-13].tolist()

# X4 - Stem specific density (SSD) or wood density (stem dry mass per stem fresh volume)
# X11 - Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA)
# X18 - Plant height
# X26 - Seed dry mass
# X50 - Leaf nitrogen (N) content per leaf area
# X3112 - Leaf area (in case of compound leaves: leaf, undefined if petiole in- or excluded) 


# %% [markdown]
# ### Folds

# %%
if config.FOLDS > 1:
    gkf = GroupKFold(n_splits=config.FOLDS)
    for fold, (train_index, valid_index) in enumerate(gkf.split(df, df[target_cols], df['id'])):
        df.loc[valid_index, "fold"] = int(fold)

# %% [markdown]
# ### Filter huge outliers

# %%
# Mean of stem specific density is larger than 0
df = df[df['X4_mean'] > 0]

# Mean of Leaf area per leaf dry mass is smaller than 250
df = df[df['X11_mean'] < 250]

# Mean of plant height is smaller than 100 m
df = df[df['X18_mean'] < 100]

# Mean of dry seed mass is smaller than 100000 mg
df = df[df['X26_mean'] < 100000]

# Mean of nitrogen content is smaller than 100
df = df[df['X50_mean'] < 100]

# Mean of leaf area is smaller than 100000 mm2
df = df[df['X3112_mean'] < 100000]

print(df.shape)

# %% [markdown]
# ### Adding classes 

# %%
y_scaler = StandardScaler()
df[target_cols] = np.log(df[target_cols])
df[target_cols] = y_scaler.fit_transform(df[target_cols])

classes = df[target_cols]
classes = classes.drop_duplicates(ignore_index=True)
classes['class_labels'] = classes.apply(lambda x: x.name, axis=1)

# df = df.merge(classes, on=target_cols, how='outer')

num_classes = len(target_cols)
print(num_classes)

# %% [markdown]
# ### Normalize labels and save scaler for test

# %%

#img_means,img_stds = calculate_imgs_mean_std(df['image_files'])
img_means, img_stds = [ 85.91545542, 115.27089263, 115.54142482],[58.83891828, 56.6869845, 59.33508381]
print(img_means)
print(img_stds)

# %% [markdown]
# ## Dataset

# %%
class PlantDataset(Dataset):
    def __init__(self, paths, features, labels=None, mode='train', augment="both"):
        self.paths = paths
        self.features = features
        self.labels = labels
        self.mode = mode
        self.augment = augment

        if self.mode == 'train':
            self.transform = self.build_augmenter(self.augment)
        else:
            self.transform = self.validation_transformer()

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        path = self.paths[idx]
        feature = self.features[idx]

        # Read and decode image
        image = self.decode_image(path)

        transformed = self.transform(image=image)
        image = transformed['image']            


        if self.labels is not None:
            label = torch.tensor(self.labels[idx])
            # class_label = torch.tensor(self.class_labels[idx])

            return {'images': image, 'features': feature}, (label)
        else:
            return {'images': image, 'features': feature}
     
    def decode_image(self, path):
        image = Image.open(path)
        image = np.array(image)
        return image
    

    def build_augmenter(self, augment="both"):
        mean = [i / 255 for i in img_means]
        std = [i / 255 for i in img_stds]
        
        base = [A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                A.ToFloat(),
                A.Normalize(mean=mean, std=std, max_pixel_value=1),
                ToTensorV2(),
            ]
        flip = [A.HorizontalFlip(p=0.5)]
        crop = [A.RandomSizedCrop(
                [int(0.85*config.IMG_SIZE), config.IMG_SIZE],
                config.IMG_SIZE, config.IMG_SIZE, w2h_ratio=1.0, p=0.75)]
        brightness = [A.RandomBrightnessContrast(brightness_limit=(-0.2, 0.2),
                                        contrast_limit=(-0.2, 0.2), p=0.2)]
        
        if augment == "none":
            transform = A.Compose(base)
        elif augment == "both":
            transform = A.Compose(flip + crop + brightness + base)
        else:
            transform = A.Compose(base)

        return transform

    def validation_transformer(self):
        mean = [i / 255 for i in img_means]
        std = [i / 255 for i in img_stds]
        
        return A.Compose([
                A.Resize(config.IMG_SIZE, config.IMG_SIZE),
                A.ToFloat(),
                A.Normalize(mean=mean, std=std, max_pixel_value=1),
                ToTensorV2(),
            ])


# %% [markdown]
# # Model

# %%
class PlantViTModel(nn.Module):
    def __init__(self, model_name):
        super().__init__()
        self.model_name = model_name
        
        self.model = timm.create_model(self.model_name,
                                       pretrained=True, 
                                       num_classes=num_classes)
        
    def forward(self, x):
        
        out = self.model(x)
        
        return out


# %%
R2 = torchmetrics.regression.R2Score(num_outputs=6, multioutput='uniform_average').to(device)

# %% [markdown]
# ## Train and Validation Functions

# %%
def train_epoch(train_loader, model, optimizer, epoch, scheduler, loss_fn):
    model.train()
    R2.reset()
    
    scaler = torch.cuda.amp.GradScaler(enabled=config.AMP)
    losses = AverageMeter()
    global_step = 0
    
    for X, y in train_loader:
        X = X['images'].to(device)
        # y_features = y[0]
        y = y.to(device)
        batch_size = y.size(0)
        
        optimizer.zero_grad()
        
        with torch.cuda.amp.autocast(enabled=config.AMP):
            X = X.half()
            y_preds = model(X)
            loss = loss_fn(y_preds, y)
#                 y_pred_classes = np.argmax(y_preds.detach().cpu(), axis=1)
#                 y_pred_values = [classes[classes['class_labels'] == y_p_c.item()][target_cols] 
#                                  for y_p_c in y_pred_classes]
        losses.update(loss.item(), batch_size)
        R2.update(y_preds, y)
#             R2.update(np.array(y_pred_values).squeeze(), y_features)
        
        scaler.scale(loss).backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), config.MAX_GRAD_NORM)
        scaler.step(optimizer)
        scaler.update()
        optimizer.zero_grad()
        scheduler.step()

    print(f'Train loss: {losses.avg:.4f}, R2: {R2.compute().item():.4f}')
    return losses.avg, R2.compute().item()


def valid_epoch(valid_loader, model, loss_fn):  
    model.eval()
    R2.reset()
        
    losses = AverageMeter()
    preds = np.empty((0,6))
    start = end = time.time()
    for X, y in valid_loader:
        X = X['images'].to(device)
        # y_features = y[0]
        y = y.to(device)
        batch_size = y.size(0)
        with torch.no_grad():
            X = X.float()
            y_preds = model(X)
            loss = loss_fn(y_preds, y)
#                 y_pred_classes = np.argmax(y_preds.detach().cpu(), axis=1)
#                 y_pred_values = [classes[classes['class_labels'] == y_p_c.item()][target_cols] 
#                                  for y_p_c in y_pred_classes]
        losses.update(loss.item(), batch_size)
        R2.update(y_preds, y)

        preds = np.vstack((preds, y_preds.to('cpu').numpy()))
#             R2.update(np.array(y_pred_values).squeeze(), y_features)


#             preds = np.vstack((preds, np.array(y_pred_values).squeeze()))
        end = time.time()

    print(f'Val loss: {losses.avg:.4f}, R2: {R2.compute().item():.4f}')
    return losses.avg, preds, R2.compute().item()


# %%
def train_loop(cfg):
    train_folds, valid_folds = train_test_split(df, test_size=0.2, random_state=42)
        
    # Normalize features
    scaler = StandardScaler()
    train_features = scaler.fit_transform(train_folds[FEATURE_COLS].values)
    valid_features = scaler.transform(valid_folds[FEATURE_COLS].values)
    
    # Extract file paths, features, labels, and fold information for train and validation sets
    train_paths = train_folds.image_files.values
    train_labels = train_folds[target_cols].values
    # train_class_labels = train_folds['class_labels'].values
    
    print(train_labels.shape)
    # print(train_class_labels.shape)

    valid_paths = valid_folds.image_files.values
    valid_labels = valid_folds[target_cols].values
    # valid_class_labels = valid_folds['class_labels'].values
    
    train_dataset = PlantDataset(train_paths, train_features, train_labels,
                         mode='train', augment=cfg['augment'])
    
    valid_dataset = PlantDataset(valid_paths, valid_features, valid_labels,
                         mode='valid', augment='none')
    
    train_loader = DataLoader(train_dataset,
                              batch_size=cfg['batch_size'],
                              shuffle=True,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=True)
    valid_loader = DataLoader(valid_dataset,
                              batch_size=cfg['batch_size'],
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)
    
    model = PlantViTModel(cfg['model_name'])
    model.to(device)
    model = nn.DataParallel(model)
    
    if cfg['optimizer'] == 'adamw':
      optimizer = torch.optim.AdamW(model.parameters(), lr=0.1, weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'adam':
      optimizer = torch.optim.Adam(model.parameters(), lr=0.1, weight_decay=cfg['weight_decay'])
    elif cfg['optimizer'] == 'sgd':
      optimizer = torch.optim.SGD(model.parameters(), lr=0.1, weight_decay=cfg['weight_decay'], momentum=cfg['momentum'])

    scheduler = OneCycleLR(
        optimizer,
        max_lr=cfg['max_lr'],
        epochs=int(cfg['epochs']),
        steps_per_epoch=len(train_loader),
        pct_start=cfg['pct_start'],
        anneal_strategy="cos",
        div_factor=cfg['div_factor'],
        final_div_factor=cfg['final_div_factor'],
    )
    
#     loss_fn = MegatronLoss(classes)
    loss_fn = nn.SmoothL1Loss(beta=cfg['beta'])

    
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
    train_r2s = []
    val_r2s = []
    
    for epoch in range(int(cfg['epochs'])):
        avg_train_loss, train_r2 = train_epoch(train_loader, model, optimizer, epoch, scheduler, loss_fn)
        avg_val_loss, preds, val_r2 = valid_epoch(valid_loader, model, loss_fn)
        train_losses.append(avg_train_loss)
        val_losses.append(avg_val_loss)
        train_r2s.append(train_r2)
        val_r2s.append(val_r2)
        
        with tempfile.TemporaryDirectory() as temp_checkpoint_dir:
            path = os.path.join(temp_checkpoint_dir, "checkpoint.pt")
            torch.save(
                (model.state_dict(), optimizer.state_dict()), path
            )
            checkpoint = Checkpoint.from_directory(temp_checkpoint_dir)
            train.report(
                {"loss": avg_val_loss,
                "train_loss": avg_train_loss,
                "train_r2": train_r2,
                "val_r2": val_r2,},
                checkpoint=checkpoint,
            )

# %%
def main(num_samples=100, max_num_epochs=10, gpus_per_trial=1):
  config = {
    'model_name': tune.choice(['swin_large_patch4_window12_384.ms_in22k_ft_in1k',
                                'vit_large_patch16_384.augreg_in21k_ft_in1k', 
                                'efficientnet_b2',
                                'efficientnet_b4']),
    'max_lr': tune.loguniform(1e-5, 0.1),
    'weight_decay': tune.loguniform(1e-3, 0.1),
    'batch_size': tune.choice([4,8,16,32]),
    'epochs': tune.quniform(4, 12, 1),
    'optimizer': tune.choice(['adam', 'adamw', 'sgd']),
    'augment': tune.choice(['none', 'both']),
    'pct_start': tune.uniform(0.1, 0.4),
    'div_factor': tune.uniform(10, 100),
    'final_div_factor': tune.uniform(10,100),
    'momentum': tune.uniform(0.1, 0.9),
    'beta': tune.uniform(0.1, 2.0)
  }
  algo = OptunaSearch()
  algo = ConcurrencyLimiter(algo, max_concurrent=2)

  tuner = tune.Tuner(
    tune.with_resources(
      tune.with_parameters(train_loop),
      resources={"cpu": 2, "gpu": gpus_per_trial}
    ),
    tune_config=tune.TuneConfig(
      metric="val_r2",
      mode="max",
      search_alg=algo,
      num_samples=num_samples
    ),
    run_config=train.RunConfig(
      storage_path=Path(f"{paths.PONYHOME}/planttraits/results").resolve(), 
      name=NAME,
      log_to_file=f"{paths.PONYHOME}/planttraits/logs/{NAME}.log"
      ),
    param_space=config,
  )
  results = tuner.fit()

  best_result = results.get_best_result("val_r2", "max")
  print("Best trial config: {}".format(best_result.config))
  print("Best trial final validation r2: {}".format(
      best_result.metrics["val_r2"]))


main()
