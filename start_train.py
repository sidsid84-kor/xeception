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


import argparse

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--img_size', type=int, default=640, help='img_size')
parser.add_argument('--epoch', type=int, default=200, help='epoch')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--test_ratio', type=float, default=0.2, help='test_ratio')
parser.add_argument('--img_dir', type=str, default='./data/images2', help='img_dri')
parser.add_argument('--csv_path', type=str, default='dataset.csv', help='csv_path')
parser.add_argument('--train_name', type=str, default="train_", help='train name')
parser.add_argument('--weight', type=str, default="None", help='pretrained_weight_path')

args = parser.parse_args()

IMG_SIZE = args.img_size
EPOCH = args.epoch
BATCH_SIZE = args.batch_size
TRAIN_RATIO = args.test_ratio
IMG_DIR = args.img_dir
CSV_PATH = args.csv_path
SAVE_FOLDER_NAME = args.train_name #folder name - > models/train_0/ save weights and result
weight_path = args.weight

def create_directory():
    i = 1
    while True:
        dir_name = os.path.join('models/'+SAVE_FOLDER_NAME+ str(i) +'/')
        if not os.path.exists(dir_name):
            os.makedirs(dir_name)
            os.makedirs(dir_name+'/result')
            return dir_name
            break
        i += 1

save_path = create_directory()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)

torch.cuda.reset_max_memory_allocated(device=None)
torch.cuda.empty_cache()


# define transformation
transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(IMG_SIZE)
])


train_df, val_df, NUM_CLS, cls_list = get_data_from_csv(csv_path=CSV_PATH,img_dir=IMG_DIR, train_ratio=TRAIN_RATIO, randoms_state=42)


train_set = CustomDataset(train_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list ,img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE))
train_set.transforms = transformation

val_set = CustomDataset(val_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list, img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE))
val_set.transforms = transformation


model = Xception(num_classes=NUM_CLS)

if weight_path != "None":
    model.load_state_dict(torch.load(weight_path, map_location=device))

num_devices = torch.cuda.device_count()
device_idx = []
for i in range(num_devices):
    if torch.cuda.get_device_name(i) == "NVIDIA DGX Display":
        print(f"Device is not using : {torch.cuda.get_device_name(i)}")
    else:
        device_idx.append(i)

if torch.cuda.device_count() > 1:
    num_device = torch.cuda.device_count()
    print("Let's use",num_device, "GPUs!")
    if torch.cuda.device_count() > 4: #for GCT
        model=model.to('cuda:0')
        model = nn.DataParallel(model, device_ids=device_idx)
    else:
        model = model.to(device=device)
        model = nn.DataParallel(model)



train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4*num_device)
val_loader = DataLoader(train_set, batch_size=int(BATCH_SIZE//num_device), num_workers=4)


# define loss function, optimizer, lr_scheduler
loss_func = nn.MultiLabelSoftMarginLoss()
opt = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=50)

params_train = {
    'num_epochs':EPOCH,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_loader,
    'val_dl':val_loader,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':save_path,
}

summary(model, (3, IMG_SIZE, IMG_SIZE), device=device.type)

traind_model, loss_hist, metric_hist, metric_cls_hist = train_val(model, device, params_train)

