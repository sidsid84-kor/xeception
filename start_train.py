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



from train import *

from dataset import *


import argparse

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--model', type=str, default="xeception", help='select model')
parser.add_argument('--img_size', type=int, default=640, help='img_size')
parser.add_argument('--epoch', type=int, default=200, help='epoch')
parser.add_argument('--batch_size', type=int, default=32, help='batch_size')
parser.add_argument('--test_ratio', type=float, default=0.2, help='test_ratio')
parser.add_argument('--img_dir', type=str, default='./data/images2', help='img_dri')
parser.add_argument('--csv_path', type=str, default='dataset.csv', help='csv_path')
parser.add_argument('--val_csv_path', type=str, default=None, help='val_csv_path')
parser.add_argument('--train_name', type=str, default="train_", help='train name')
parser.add_argument('--weight', type=str, default="None", help='pretrained_weight_path')
parser.add_argument('--loss', type=str, default="multi", help='multi or softmax')


args = parser.parse_args()
SELECTED_MODEL = args.model
IMG_SIZE = args.img_size
EPOCH = args.epoch
BATCH_SIZE = args.batch_size
TRAIN_RATIO = args.test_ratio
IMG_DIR = args.img_dir
CSV_PATH = args.csv_path
VAL_PATH = args.val_csv_path
SAVE_FOLDER_NAME = args.train_name #folder name - > models/train_0/ save weights and result
LOSS_MODE = args.loss
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


train_df, val_df, NUM_CLS, cls_list = get_data_from_csv(csv_path=CSV_PATH,img_dir=IMG_DIR, train_ratio=TRAIN_RATIO, randoms_state=42, val_csv_path=VAL_PATH)


train_set = CustomDataset(train_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list ,img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE))
train_set.transforms = transformation

val_set = CustomDataset(val_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list, img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE))
val_set.transforms = transformation

#################################################모델선언!
if SELECTED_MODEL == 'xeception':
    from xeception import *
    model = Xception(num_classes=NUM_CLS)

elif SELECTED_MODEL == 'googlenetv4':
    from googlenetv4 import *
    model = InceptionV4(num_classes=NUM_CLS)

elif SELECTED_MODEL == 'visionT':
    from ViT import ViT
    model = ViT(num_classes=NUM_CLS)
    
else:
    print('select model in list - xeception , googlenetv4 ,  visionT')

print(f'train with {SELECTED_MODEL}')
#######################################가중치 이어서 돌릴경우임.
if weight_path != "None":
    model.load_state_dict(torch.load(weight_path, map_location=device))

########################gpu개수 세고.. 병렬로 자동으로...뭐...기타..#
num_device = torch.cuda.device_count()
device_idx = []
for i in range(num_device):
    if torch.cuda.get_device_name(i) == "NVIDIA DGX Display":
        print(f"Device is not using : {torch.cuda.get_device_name(i)}")
    else:
        device_idx.append(i)

if torch.cuda.device_count() > 1:
    print("Let's use",num_device, "GPUs!")
    if torch.cuda.device_count() > 4: #for GCT
        model=model.to('cuda:0')
        model = nn.DataParallel(model, device_ids=device_idx)
    else:
        model = model.to(device=device)
        model = nn.DataParallel(model)
else:
    model = model.to(device=device)

def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    
    # Find the maximum target length
    max_len = max([len(t) for t in targets])
    
    # Pad each target to the maximum length
    targets_padded = [torch.cat([torch.tensor(t), torch.zeros(max_len - len(t))]) for t in targets]
    
    targets = torch.stack(targets_padded, 0)
    return images, targets



############################윈도우에서는 워커 주면안됨
if os.name == "nt":
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_set, batch_size=int(BATCH_SIZE//num_device))
else:
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_set, batch_size=int(BATCH_SIZE//num_device), collate_fn=collate_fn, num_workers=4)
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4*num_device)
    # val_loader = DataLoader(val_set, batch_size=int(BATCH_SIZE//num_device), num_workers=4)


# define loss function, optimizer, lr_scheduler
if LOSS_MODE == 'multi':
    loss_func = nn.MultiLabelSoftMarginLoss()
elif LOSS_MODE == 'softmax':
    loss_func = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=0.001)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=5)

params_train = {
    'num_epochs':EPOCH,
    'optimizer':opt,
    'loss_func':loss_func,
    'train_dl':train_loader,
    'val_dl':val_loader,
    'sanity_check':False,
    'lr_scheduler':lr_scheduler,
    'path2weights':save_path,
    'loss_mode' : LOSS_MODE,
}

summary(model, (3, IMG_SIZE, IMG_SIZE), device=device.type)

traind_model, loss_hist, metric_hist, metric_cls_hist = train_val(model, device, params_train)

