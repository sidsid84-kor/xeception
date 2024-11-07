import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
from torch.multiprocessing import spawn

# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import os
import sys

# display images
from torchvision import utils
import matplotlib.pyplot as plt

# utils
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from torchinfo import summary

#train
from torch import optim
from torch.optim.lr_scheduler import ReduceLROnPlateau



from train import *

from dataset import *


import argparse

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
        
def collate_fn(batch):
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    max_len = max(len(t) for t in targets)
    targets_padded = [torch.cat([torch.tensor(t), torch.zeros(max_len - len(t))]) for t in targets]
    targets = torch.stack(targets_padded, 0)

    return images, targets
    
def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12355'
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)
    
def train_ddp(rank, world_size):
    setup(rank, world_size)

    # 모델 초기화 및 DDP 적용
    if SELECTED_MODEL == 'xeception':
        model = Xception(num_classes=NUM_CLS)

    elif SELECTED_MODEL == 'googlenetv4':
        model = InceptionV4(num_classes=NUM_CLS, dropout_prob=DropOut_RATE)

    elif SELECTED_MODEL == 'visionT':
        model = ViT(num_classes=NUM_CLS)

    elif SELECTED_MODEL == 'efficientnet':
        model = EfficientNet.from_name('efficientnet-c1', num_classes = NUM_CLS)

    elif SELECTED_MODEL == 'th_googlenetv4':
        model = TH_InceptionV4(num_classes=NUM_CLS)
        

    elif SELECTED_MODEL == 'th_efficientnet':
        model = TH_EfficientNet.from_name('efficientnet-c1', num_classes = NUM_CLS)


    print(f'train with {SELECTED_MODEL}')
    
    model = model.cuda(rank)
    model = nn.parallel.DistributedDataParallel(model, device_ids=[rank])
    # 데이터 로더 준비
    train_sampler = DistributedSampler(dataset=train_set, num_replicas=len(device_idx), rank=rank)
    train_loader = DataLoader(dataset=train_set, batch_size=BATCH_SIZE, sampler=train_sampler)
    val_loader = DataLoader(dataset=val_set, batch_size=BATCH_SIZE)

    # 트레이닝 로직
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

    traind_model, loss_hist, metric_hist, metric_cls_hist = train_val(model, device, params_train)
    dist.destroy_process_group()

parser = argparse.ArgumentParser(description='parameters')
parser.add_argument('--config', type=str, default="./parameters/base.txt", help='configed txt filepath')

args = parser.parse_args()

with open(args.config) as f:
    lines = [line.strip() for line in f.readlines()]
    SELECTED_MODEL = lines[0].split(">>")[1]
    IMG_SIZE = int(lines[1].split(">>")[1])
    EPOCH = int(lines[2].split(">>")[1])
    BATCH_SIZE = int(lines[3].split(">>")[1])
    TRAIN_RATIO = float(lines[4].split(">>")[1])
    IMG_DIR = lines[5].split(">>")[1]
    CSV_PATH = lines[6].split(">>")[1]
    VAL_PATH = lines[7].split(">>")[1]
    SAVE_FOLDER_NAME = lines[8].split(">>")[1]
    weight_path = lines[9].split(">>")[1]
    LOSS_MODE = lines[10].split(">>")[1]
    Learning_RATE = float(lines[11].split(">>")[1])
    LR_patience = int(lines[12].split(">>")[1])
    DropOut_RATE = float(lines[13].split(">>")[1])



train_df = pd.read_csv(CSV_PATH)
NUM_CLS = len(train_df.columns) - 1  # because, it is multi-label

#################################################모델선언!
model_list = ['xeception', 'googlenetv4','visionT', 'efficientnet', 'th_googlenetv4', 'polar_gru', 'polar_lstm', 'polar_transformer']
if SELECTED_MODEL not in model_list:
    print("해당 모델은 없음")
    print(f"{model_list} 에서 선택해야함")
    sys.exit()

if SELECTED_MODEL == 'xeception':
    from xeception import *
    model = Xception(num_classes=NUM_CLS)

elif SELECTED_MODEL == 'googlenetv4':
    from googlenetv4 import *
    model = InceptionV4(num_classes=NUM_CLS, dropout_prob=DropOut_RATE)

elif SELECTED_MODEL == 'visionT':
    from ViT import ViT
    model = ViT(num_classes=NUM_CLS)

elif SELECTED_MODEL == 'efficientnet':
    from efficientnet import EfficientNet
    model = EfficientNet.from_name('efficientnet-c1', num_classes = NUM_CLS)

elif SELECTED_MODEL == 'th_googlenetv4':
    from TH_googlenet import *
    model = TH_InceptionV4(num_classes=NUM_CLS)
    

elif SELECTED_MODEL == 'th_efficientnet':
    from TH_efficientnet import *
    model = TH_EfficientNet.from_name('efficientnet-c1', num_classes = NUM_CLS)

elif SELECTED_MODEL == 'polar_gru':
    transformation = transforms.Compose([
                    transforms.ToTensor(),
    ])
    train_set = CustomDataset(train_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list ,img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE), polar_tranform=True)
    train_set.transforms = transformation

    val_set = CustomDataset(val_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list, img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE), polar_tranform=True)
    val_set.transforms = transformation

    sample_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    custom_size = True
    from polar import *
    model = CNN_GRU_Model(num_classes = NUM_CLS)
    # DataLoader를 통해 모델에 한 번 데이터를 통과시켜 초기화
    for images, _ in sample_loader:
        # 데이터를 모델에 통과시켜 초기화
        print(images.shape)
        break  # 하나의 배치만 처리하고 반복문 탈출

elif SELECTED_MODEL == 'polar_lstm':
    transformation = transforms.Compose([
                    transforms.ToTensor(),
    ])
    train_set = CustomDataset(train_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list ,img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE), polar_tranform=True)
    train_set.transforms = transformation

    val_set = CustomDataset(val_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list, img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE), polar_tranform=True)
    val_set.transforms = transformation

    sample_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    custom_size = True
    from polar import *
    model = CNN_LSTM_Model(num_classes = NUM_CLS)
    # DataLoader를 통해 모델에 한 번 데이터를 통과시켜 초기화
    for images, _ in sample_loader:
        # 데이터를 모델에 통과시켜 초기화
        print(images.shape)
        break  # 하나의 배치만 처리하고 반복문 탈출


elif SELECTED_MODEL == 'polar_transformer':
    transformation = transforms.Compose([
                    transforms.ToTensor(),
    ])
    train_set = CustomDataset(train_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list ,img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE), polar_tranform=True)
    train_set.transforms = transformation

    val_set = CustomDataset(val_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list, img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE), polar_tranform=True)
    val_set.transforms = transformation

    sample_loader = DataLoader(train_set, batch_size=1, shuffle=True)
    custom_size = True
    from polar import *
    model = CNN_TRANS_Model(num_classes = NUM_CLS)
    # DataLoader를 통해 모델에 한 번 데이터를 통과시켜 초기화
    for images, _ in sample_loader:
        # 데이터를 모델에 통과시켜 초기화
        print(images.shape)
        break  # 하나의 배치만 처리하고 반복문 탈출
    
#######################################가중치 이어서 돌릴경우임.
if weight_path != "None":
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)
    
    
# define loss function, optimizer, lr_scheduler
if LOSS_MODE == 'multi':
    loss_func = nn.MultiLabelSoftMarginLoss()
elif LOSS_MODE == 'softmax':
    loss_func = nn.CrossEntropyLoss()
opt = optim.Adam(model.parameters(), lr=Learning_RATE)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=LR_patience)

if __name__ == '__main__':

    print('SELECTED_MODEL',SELECTED_MODEL)
    print('IMG_SIZE',IMG_SIZE)
    print('EPOCH',EPOCH)
    print('BATCH_SIZE',BATCH_SIZE)
    print('TRAIN_RATIO',TRAIN_RATIO)
    print('IMG_DIR',IMG_DIR)
    print('CSV_PATH',CSV_PATH)
    print('VAL_PATH',VAL_PATH)
    print('SAVE_FOLDER_NAME',SAVE_FOLDER_NAME)
    print('weight_path',weight_path)
    print('loss_MODE',LOSS_MODE)
    print('Learning_RATE',Learning_RATE)
    print('LR_patience',LR_patience)
    print('DropOut_RATE',DropOut_RATE)

    save_path = create_directory()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)

    torch.cuda.reset_max_memory_allocated(device=None)
    torch.cuda.empty_cache()

    custom_size = False

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


    print(f'train with {SELECTED_MODEL}')
    sample_loader = DataLoader(train_set, batch_size=1, collate_fn=collate_fn)
    images, labels = next(iter(sample_loader)) 



    if custom_size:
        print(images.shape)
        #summary(model, (images.size(1), images.size(2), images.size(3)), device=device.type)
    else:
        summary(model, input_size=(1, 3, IMG_SIZE, IMG_SIZE))
    print('start')
    
    num_devices = torch.cuda.device_count()
    device_idx = [i for i in range(num_devices) if torch.cuda.get_device_name(i) != "NVIDIA DGX Display"]
    world_size = len(device_idx)
    spawn(train_ddp, args=(world_size,), nprocs=world_size, join=True)