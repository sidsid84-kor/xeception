# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchsummary import summary

#train
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau


from xeception_2x1ch import *
from xeception import *
from train import *

from dataset import *



IMG_SIZE = 640
BATCH_SIZE = 32
TRAIN_RATIO = 0.2


device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.cuda.reset_max_memory_allocated(device=None)
torch.cuda.empty_cache()


# define transformation
transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(IMG_SIZE)
])



img_dir = './data/images2'

train_df, val_df, NUM_CLS, cls_list = get_data_from_csv(csv_path='dataset.csv',img_dir=img_dir, train_ratio=TRAIN_RATIO, randoms_state=42)


train_set = CustomDataset(train_df,num_classes=NUM_CLS, image_dir='./data/images2', class_list= cls_list ,img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE))
train_set.transforms = transformation

val_set = CustomDataset(val_df,num_classes=NUM_CLS, image_dir='./data/images2', class_list= cls_list, img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE))
val_set.transforms = transformation


model = Xception2x1ch(num_classes=NUM_CLS).to(device)
if torch.cuda.device_count() > 1:
    num_device = torch.cuda.device_count()
    print("Let's use",num_device, "GPUs!")
    model = nn.DataParallel(model)



train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4*num_device)
val_loader = DataLoader(train_set, batch_size=int(BATCH_SIZE//num_device), num_workers=4)




# define loss function, optimizer, lr_scheduler
loss_func = nn.MultiLabelSoftMarginLoss()
opt = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=50)

params_train = {
    'num_epochs':500,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_loader,
    'val_dl':val_loader,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':'./models/',
}

summary(model, (3, IMG_SIZE, IMG_SIZE), device=device.type)

traind_model, loss_hist, metric_hist, metric_cls_hist = train_val(model, device, params_train)



lossdf = pd.DataFrame(loss_hist)
accdf = pd.DataFrame(metric_hist)
acc_clsdf = pd.DataFrame(metric_cls_hist)

lossdf.to_csv('loss.csv')
accdf.to_csv('acc.csv')
acc_clsdf.to_csv('cls_acc.csv')