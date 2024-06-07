# %%
from PIL import Image
import albumentations as A
import gc
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
from sklearn.model_selection import GroupKFold
from sklearn.preprocessing import StandardScaler
import torchmetrics
from sklearn.metrics import r2_score
import xgboost as xgb
import cupy as cp

from albumentations.pytorch import ToTensorV2
from glob import glob
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from hyperopt import hp, atpe, Trials, STATUS_OK, STATUS_FAIL
from hyperopt.pyll.base import scope
from hyperopt.early_stop import no_progress_loss
from hyperopt import fmin

from functools import partial
import warnings
np.warnings = warnings

print(list(range(torch.cuda.device_count())))
device = torch.device("cuda:0")
cp.cuda.Device(0).use()
print('Using ', device)
print(torch.cuda.get_device_name(device))

# %% [markdown]
# # Config

# %%
class config:
    DEVICE = 'cuda'
    AMP = True
    BATCH_SIZE =32
    FOLDS = 5
    VAL_SIZE = .2   # If FOLDS=1 
    MAX_GRAD_NORM = 1e7
    MODEL = "vit_large_patch16_384.augreg_in21k_ft_in1k"
    NUM_WORKERS = 0 # multiprocessing.cpu_count()
    PRINT_FREQ = 20
    SEED = 20
    IMG_SIZE = 384
    
class paths:
    OUTPUT_DIR = './output'
    DATASET = './data/'
    IMAGES = './data/train_images/'
    IMAGES_TEST = './data/test_images/'
    
model_weights = [x for x in glob("./data/checkpoint.pt")]
trained_model = "./data/checkpoint.pt"

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
train_df = pd.read_csv(paths.DATASET + 'train.csv')
train_df['image_files'] = train_df['id'].apply(lambda x: paths.IMAGES +str(x)+'.jpeg')
train_df.loc[:, aux_class_names] = train_df.loc[:, aux_class_names].fillna(-1)
FEATURE_COLS = train_df.columns[1:-13].tolist()

test_df = pd.read_csv(paths.DATASET + 'test.csv')
test_df['image_files'] = test_df['id'].apply(lambda x: paths.IMAGES_TEST +str(x)+'.jpeg')

# X4 - Stem specific density (SSD) or wood density (stem dry mass per stem fresh volume)
# X11 - Leaf area per leaf dry mass (specific leaf area, SLA or 1/LMA)
# X18 - Plant height
# X26 - Seed dry mass
# X50 - Leaf nitrogen (N) content per leaf area
# X3112 - Leaf area (in case of compound leaves: leaf, undefined if petiole in- or excluded) 


# %% [markdown]
# ### Filter huge outliers

# %%
# Mean of stem specific density is larger than 0
train_df = train_df[train_df['X4_mean'] > 0]

# Mean of Leaf area per leaf dry mass is smaller than 250
train_df = train_df[train_df['X11_mean'] < 250]

# Mean of plant height is smaller than 100 m
train_df = train_df[train_df['X18_mean'] < 100]

# Mean of dry seed mass is smaller than 100000 mg
train_df = train_df[train_df['X26_mean'] < 100000]

# Mean of nitrogen content is smaller than 100
train_df = train_df[train_df['X50_mean'] < 100]

# Mean of leaf area is smaller than 100000 mm2
train_df = train_df[train_df['X3112_mean'] < 100000]

print(train_df.shape)

# %% [markdown]
# ### Adding classes 

# %%
y_scaler = StandardScaler()
train_df[target_cols] = np.log(train_df[target_cols])
train_df[target_cols] = y_scaler.fit_transform(train_df[target_cols])

img_means, img_stds = [ 85.91545542, 115.27089263, 115.54142482],[58.83891828, 56.6869845, 59.33508381]

num_classes = len(target_cols)
print(num_classes)

# %%
MEAN = np.array(img_means)
STD = np.array(img_stds)

# %% [markdown]
# ## Dataset

# %%
class PlantDataset(Dataset):
    def __init__(self, paths, features, labels=None, mode="test", augment="none"):
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
    
class PlantViTModel(nn.Module):
    def __init__(self, ):
        super().__init__()
       
        self.model = timm.create_model(config.MODEL,
                                       pretrained=True, 
                                       num_classes=num_classes)
        
    def forward(self, x):
        
        out = self.model(x)
        
        return out

# %%
class HeadlessPlantViTModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.full_model = PlantViTModel()
        self.full_model = nn.DataParallel(self.full_model)
        
        self.checkpoint = torch.load(trained_model)
        self.full_model.load_state_dict(self.checkpoint[0])
        
        self.model = self.full_model.module.model
        
        self.classifier = self.model.head
        
        self.features = self.model
        self.features.head = nn.Identity()
        
    def forward(self, x):
        features = self.features(x)
        preds = self.classifier(features)
        
        return features, preds
    
# %% [markdown]
# ## Loss function

# %%
R2 = torchmetrics.regression.R2Score(num_outputs=6, multioutput='uniform_average')

# %% [markdown]
# ## Train and Validation Functions

# %%
def inference_function(test_loader, model, device):
    model.eval()

    all_tabular = []
    all_features = []
    all_preds = []
    all_labels = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, (X,y) in enumerate(tqdm_test_loader):
            images = X['images'].to(device)
            tabular = X['features']
            labels = y.numpy()
            
            with torch.no_grad():
                y_features, y_preds_scaled = model(images)
                    
            all_tabular.extend(tabular)
            all_features.extend(y_features.to('cpu').numpy())
            all_preds.extend(y_preds_scaled.to('cpu').numpy())
            all_labels.extend(labels)

    return all_tabular, all_features, all_preds, all_labels

# %%
def train_loop(df):        
    # Extract file paths, features, labels, and fold information for train and validation sets
    train_features = df[FEATURE_COLS].values
    
    train_paths = df.image_files.values
    train_labels = df[target_cols].values
    
    train_dataset = PlantDataset(train_paths, train_features, train_labels,
                         mode='test')
    
    train_loader = DataLoader(train_dataset,
                              batch_size=config.BATCH_SIZE,
                              shuffle=False,
                              num_workers=config.NUM_WORKERS, pin_memory=True, drop_last=False)
        
    model_headless = HeadlessPlantViTModel()
    model_headless.to(device)
    
    all_tabular, all_features, all_preds, all_labels = inference_function(train_loader, model_headless, device)    


    torch.cuda.empty_cache()
    gc.collect()
    
    return all_tabular, all_features, all_preds, all_labels

# %%
torch.cuda.empty_cache()
gc.collect()

all_tabular, all_features, all_preds, all_labels = train_loop(train_df)

# %%
all_tabular, all_features, all_preds, all_labels = np.array(all_tabular), np.array(all_features), np.array(all_preds), np.array(all_labels)

print(all_tabular.shape, all_features.shape, all_preds.shape, all_labels.shape)

tab_feat = np.concatenate((all_tabular,all_features), axis=1)
tab_pred = np.concatenate((all_tabular,all_preds), axis=1)
tab_featpred = np.concatenate((all_tabular,all_features,all_preds), axis=1)

df_tf = pd.DataFrame(tab_feat)
df_tp = pd.DataFrame(tab_pred)
df_tfp = pd.DataFrame(tab_featpred)
df2 = pd.DataFrame(all_labels, columns=target_cols)

classes = df2.copy()
classes = classes.drop_duplicates(ignore_index=True)
classes['class_labels'] = classes.apply(lambda x: x.name, axis=1)

feature_df = pd.concat([df_tf, df2], axis=1).merge(classes, "outer", target_cols)
predict_df = pd.concat([df_tp, df2], axis=1).merge(classes, "outer", target_cols)
featpred_df = pd.concat([df_tfp, df2], axis=1).merge(classes, "outer", target_cols)

print(feature_df.shape, predict_df.shape, featpred_df.shape)


# %%
if config.FOLDS > 1:
    gkf = GroupKFold(n_splits=config.FOLDS)
    for fold, (train_index, valid_index) in enumerate(gkf.split(featpred_df, featpred_df[target_cols], featpred_df['class_labels'])):
        feature_df.loc[valid_index, "fold"] = int(fold)
        predict_df.loc[valid_index, "fold"] = int(fold)
        featpred_df.loc[valid_index, "fold"] = int(fold)

# %%
def optimize(params):
    try:
        model= xgb.XGBRegressor(random_state=0,
                            multi_strategy="one_output_per_tree",
                            tree_method = "hist",
                            booster = "gbtree",
                            objective = "reg:squarederror",
                            eval_metric = "rmse",
                            device=device, 
                            **params)
        
        if params['dataset'] == "feature":
            X = feature_df
        elif params['dataset'] == "prediction":
            X = predict_df
        elif params['dataset'] == "feature+prediction":
            X = featpred_df
        
        R2 = []
        for i in range(config.FOLDS):
            X_train=X[X.fold != i].drop(columns=['fold', 'class_labels']).drop(columns=target_cols)
            y_train=X[X.fold !=i].drop(columns=['fold', 'class_labels'])[target_cols]
            X_test=X[X.fold == i].drop(columns=['fold', 'class_labels']).drop(columns=target_cols)
            y_test=X[X.fold==i].drop(columns=['fold', 'class_labels'])[target_cols]
            
            model.fit(cp.array(X_train), cp.array(y_train)) # I'm using cupy to move data from CPU to GPU
            y_preds = model.predict(cp.array(X_test))
            r2 = r2_score(cp.asnumpy(y_test), cp.asnumpy(y_preds))

            R2.append(r2)
            
        score = np.mean(R2)
        print("+++++++++++++++++++++++++")
        print("score: ", score)
        print(params)
        print("+++++++++++++++++++++++++")
        return {'loss': - score, 'status': STATUS_OK, 'accuracy': score}
    except Exception as e:
        print(e)
        print(f'-------------------\nPARAMS - which led to unstable failed model\n{params}')
        return {'loss': 9999, 'status': STATUS_FAIL, 'accuracy': - 9999} 

    

# %%
param_space = {
 "dataset": hp.choice("dataset", ["feature", "prediction", "feature+prediction"]),
 "learning_rate": hp.uniform("learning_rate", 0.001, 0.2),
 "max_depth": scope.int(hp.quniform("max_depth", 6, 16, 1)),
 "n_estimators": scope.int(hp.quniform("n_estimators", 100, 600, 1)),
 "subsample": hp.uniform("subsample", 0.6, 0.98),
 "colsample_bytree": hp.uniform("colsample_bytree", 0.6, 0.98),
 "colsample_bynode": hp.uniform("colsample_bynode", 0.6, 0.98),
 "reg_alpha": hp.uniform("reg_alpha", 0, 15),
 "reg_lambda": hp.uniform("reg_lambda", 0, 15),
 "gamma": hp.uniform("gamma", 0, 15),
 "min_child_weight": hp.choice('min_child_weight', np.arange(0, 101).tolist()),
 }

# %%
optimization_function = partial(
 optimize
 )


trials = Trials() 

hopt = fmin( 
 fn=optimization_function,
 space=param_space,
 algo=atpe.suggest, #partial(tpe.suggest, n_startup_jobs=50), # we have large space to validate and it defaults to 20 random runs
 max_evals=250,
 trials=trials,
 early_stop_fn=no_progress_loss(25)
 )

# %%
hopt

print(hopt)

# %%
final_xgb = xgb.XGBRegressor(random_state=0,
                                    multi_strategy="one_output_per_tree",
                                    tree_method = "hist",
                                    booster = "gbtree",
                                    objective = "reg:squarederror",
                                    eval_metric = "rmse",
                                    device=device,
                                    
                                    learning_rate=hopt['learning_rate'],
                                    max_depth=int(hopt['max_depth']),
                                    n_estimators=int(hopt['n_estimators']),
                                    gamma= hopt['gamma'],
                                    reg_alpha=hopt['reg_alpha'],
                                    reg_lambda=hopt['reg_lambda'],
                                    subsample=hopt['subsample'],
                                    colsample_bynode= hopt['colsample_bynode'],
                                    colsample_bytree= hopt['colsample_bytree'],
                                    min_child_weight=hopt['min_child_weight'],
                                    )

if hopt['dataset'] == "feature":
    X_fit = tab_feat
elif hopt['dataset'] == "prediction":
    X_fit = tab_pred
elif hopt['dataset'] == "feature+prediction":
    X_fit = tab_featpred

final_xgb.fit(X_fit, all_labels)

# %%
def test_inference_function(test_loader, model, device):
    model.eval()
    preds = []
    with tqdm(test_loader, unit="test_batch", desc='Inference') as tqdm_test_loader:
        for step, X in enumerate(tqdm_test_loader):
            img_f = X['images'].to(device)
            tab_f = X['features']
            with torch.no_grad():
                y_pred = model(img_f)
            features = np.concatenate((y_pred.to('cpu').numpy(), tab_f),axis=1)
            preds.extend(features) 

    return np.array(preds)

# %%
predictions = []

for model_weight in model_weights:
    test_features = test_df[FEATURE_COLS].values
    test_paths = test_df.image_files.values

    
    test_dataset = PlantDataset(test_paths, test_features,
                     mode="test")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=config.NUM_WORKERS,
        pin_memory=True, drop_last=False
    )
    
    model = HeadlessPlantViTModel()

    model.to(device)
    
    test_f = test_inference_function(test_loader, model, device)
    
    torch.cuda.empty_cache()
    gc.collect()


# %%
preds = final_xgb.predict(test_f)

preds = y_scaler.inverse_transform(preds)
preds = np.exp(preds)

# %%
pred_df = test_df[["id"]].copy()
target_cols = [x.replace("_mean","") for x in target_cols]
pred_df[target_cols] = preds.tolist()

pred_df = pred_df.reindex(columns=['id','X4', 'X11', 'X18', 'X50', 'X26', 'X3112'])
pred_df.to_csv("/kaggle/working/submission.csv", index=False)
pred_df.head()
print(len(pred_df))


