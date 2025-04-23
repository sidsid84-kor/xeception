# model
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchsummary import summary


# dataset and transformation
from torchvision import datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.transforms.functional import adjust_brightness, adjust_contrast, adjust_saturation
import os
import random
import shutil
import sys

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

def get_transforms(use_horizontal_flip=False,
                   use_vertical_flip=False,
                   use_rotation=False,
                   use_color_jitter=False,
                   img_size=640,
                   brightness=0.3,
                   contrast=0.3,
                   saturation=0.3,
                   rotation_degrees=359,
                   mean=0):
    transform_list = []

    # 좌우 반전 추가
    if use_horizontal_flip:
        transform_list.append(transforms.RandomHorizontalFlip(p=0.5))

    if use_vertical_flip:
        transform_list.append(transforms.RandomVerticalFlip(p=0.5))

    # 랜덤 회전 추가
    if use_rotation:
        
        transform_list.append(transforms.RandomRotation(degrees=rotation_degrees))

    # Color Jitter 추가
    if use_color_jitter:
        transform_list.append(transforms.ColorJitter(
            brightness=brightness,
            contrast=contrast,
            saturation=saturation
        ))
    
    #이거 클래스 만들어서 매번 호출해야 바뀌는거임 이렇게 하면 안됨.
    #if use_color_jitter:
    #    # 평균 0, 표준편차 0.3으로 설정된 정규분포에서 단일 샘플 값 생성
    #    brightness_factor = torch.randn(1).item() * brightness + 1
    #    contrast_factor = torch.randn(1).item() * contrast + 1
    #    saturation_factor = torch.randn(1).item() * saturation + 1
    #
    #    
    #    transform_list.append(transforms.Lambda(lambda img: adjust_brightness(img, brightness_factor)))
    #    transform_list.append(transforms.Lambda(lambda img: adjust_contrast(img, contrast_factor)))
    #    transform_list.append(transforms.Lambda(lambda img: adjust_saturation(img, saturation_factor)))

    # 텐서 변환 및 정규화 추가
    transform_list.append(transforms.Resize(img_size))
    transform_list.append(transforms.ToTensor())

    # 변환 목록을 Compose로 반환
    return transforms.Compose(transform_list)

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
    AUGMENTATION_METHOD = lines[14].split(">>")[1]

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
print('AUMENTATION_METHOD',AUGMENTATION_METHOD)

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

shutil.copy(args.config, os.path.join(save_path, "result/parameter.txt") )

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)
os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'expandable_segments:True'
torch.cuda.reset_max_memory_allocated(device=None)
torch.cuda.empty_cache()

custom_size = False

# 증강 옵션 파싱
augmentation_options = AUGMENTATION_METHOD.split(',')

use_horizontal_flip = 'horizontal_flip' in augmentation_options
use_vertical_flip = 'vertical_flip' in augmentation_options
use_rotation = 'rotation' in augmentation_options
use_color_jitter = 'color_jitter' in augmentation_options

# define transformation
transformation = get_transforms(use_horizontal_flip=use_horizontal_flip,
                                use_vertical_flip=use_vertical_flip,
                                use_rotation=use_rotation,
                                use_color_jitter=use_color_jitter,
                                img_size=IMG_SIZE,
                                brightness=0.3,
                                contrast=0.3,
                                saturation=0.3,
                                rotation_degrees=359)

is_binary = True if LOSS_MODE == "binary" else False
train_df, val_df, NUM_CLS, cls_list = get_data_from_csv(csv_path=CSV_PATH,img_dir=IMG_DIR, train_ratio=TRAIN_RATIO, randoms_state=42, val_csv_path=VAL_PATH, is_binary=is_binary)
if LOSS_MODE == 'binary' and NUM_CLS > 1:
    print("Error: Binary 는 클래스가 1개 여야함")
    sys.exit(1)  # Exit the program with an error code


train_set = CustomDataset(train_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list ,img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE), save_image=False)
train_set.transforms = transformation

transformation_Val = get_transforms(use_horizontal_flip=False,
                                use_vertical_flip=False,
                                use_rotation=False,
                                use_color_jitter=False,
                                img_size=IMG_SIZE,
                                brightness=0.3,
                                contrast=0.3,
                                saturation=0.3,
                                rotation_degrees=359)

val_set = CustomDataset(val_df,num_classes=NUM_CLS, image_dir=IMG_DIR, class_list= cls_list, img_resize=True, img_dsize=(IMG_SIZE,IMG_SIZE))
val_set.transforms = transformation_Val

#################################################모델선언!
model_list = ['xeception', 'googlenetv4','visionT','sec_model', 'efficientnet', 'th_googlenetv4','th_efficientnet', 'polar_gru', 'polar_lstm', 'polar_transformer', 'sec_effinet']
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
    model = ViT(num_classes=NUM_CLS, img_size=IMG_SIZE)

elif SELECTED_MODEL == 'sec_model':
    from sec_model import *
    model = InceptionV4_parallel(num_classes=NUM_CLS, dropout_prob=DropOut_RATE)
    
elif SELECTED_MODEL == 'sec_effinet':
    from sec_model import *
    model = Sec_Effinet(num_classes=NUM_CLS, dropout_rate=DropOut_RATE, image_size=IMG_SIZE)

elif SELECTED_MODEL == 'efficientnet':
    from efficientnet import EfficientNet
    model = EfficientNet.from_name('efficientnet-b8', num_classes = NUM_CLS)

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


print(f'train with {SELECTED_MODEL}')
#######################################가중치 이어서 돌릴경우임.
if weight_path != "None":
    model.load_state_dict(torch.load(weight_path, map_location=device), strict=False)

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
    targets = [torch.tensor(t, dtype=torch.float32) if isinstance(t, np.ndarray) else t for t in targets]
    # Find the maximum target length
    max_len = max(len(t) for t in targets)
    
    # Pad each target to the maximum length
    targets_padded = [torch.cat([t, torch.zeros(max_len - len(t), dtype=t.dtype, device=t.device)]) for t in targets]
    
    targets = torch.stack(targets_padded, 0)
    return images, targets


############################윈도우에서는 워커 주면안됨
if os.name == "nt":
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE)
    val_loader = DataLoader(val_set, batch_size=int(BATCH_SIZE//num_device))
else:
    train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, collate_fn=collate_fn, num_workers=num_device)
    val_loader = DataLoader(val_set, batch_size=int(BATCH_SIZE), collate_fn=collate_fn, num_workers=num_device)
    # train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, num_workers=4*num_device)
    # val_loader = DataLoader(val_set, batch_size=int(BATCH_SIZE//num_device), num_workers=4)


# define loss function, optimizer, lr_scheduler
if LOSS_MODE == 'multi':
    loss_func = nn.MultiLabelSoftMarginLoss()
elif LOSS_MODE == 'softmax':
    loss_func = nn.CrossEntropyLoss()
elif LOSS_MODE == 'binary':
    loss_func = nn.BCEWithLogitsLoss()
opt = optim.Adam(model.parameters(), lr=Learning_RATE)
lr_scheduler = ReduceLROnPlateau(opt, mode='min', factor=0.1, patience=LR_patience)

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

if custom_size:
    print(images.shape)
    #summary(model, (images.size(1), images.size(2), images.size(3)), device=device.type)
else:
    summary(model, (3, IMG_SIZE, IMG_SIZE), device=device.type)

traind_model, loss_hist, metric_hist, metric_cls_hist = train_val(model, device, params_train)

