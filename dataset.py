import cv2
import torch
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import tqdm

#image broken check
def check_jpeg_eoi(file_path):
    with open(file_path, 'rb') as f:
        f.seek(-2, 2) # 파일의 끝에서 두 바이트 전으로 이동합니다.
        return f.read() == b'\xff\xd9'
    

def is_image_valid(image_path):
    try:
        img = Image.open(image_path) # 이미지를 열어봅니다.
        img.verify() # verify() 메소드는 파일이 손상되었는지 확인합니다.
        return True
    except (IOError, SyntaxError) as e:
        print('Invalid image: ', image_path, '\n'+ e) # 손상된 이미지에 대한 에러 메시지를 출력합니다.
        return False

#image validation(exist and broken file)
def validate_dataset(df, img_dir):
    count = 0
    df_bar = tqdm.tqdm(df.itertuples(), desc="validating all images", total=len(df))
    for rows in df_bar:
        if os.path.isfile(img_dir+'/'+ rows.id):
            if is_image_valid(img_dir+'/'+ rows.id) and check_jpeg_eoi(img_dir+'/'+ rows.id):
                continue
            else:
                count += 1
                df.drop(df[df['id'] == rows.id].index, inplace=True)
        else:
            count += 1
            df.drop(df[df['id'] == rows.id].index, inplace=True)
        print("Not founded images (Num) : ",count)
    return df

def get_data_from_csv(csv_path, train_ratio, img_dir, randoms_state=42, val_csv_path=None):
    ###### columns example : ['id', 'good', 'b_edge', 'burr', 'borken', 'b_bubble', 'etc', 'no_lens']
    if val_csv_path is not None:
        train_df = pd.read_csv(csv_path)
        train_df = validate_dataset(df=train_df, img_dir=img_dir)
        val_df = pd.read_csv(val_csv_path)
        val_df = validate_dataset(df=val_df, img_dir=img_dir)
    else:
        df = pd.read_csv(csv_path)
        df = validate_dataset(df=df,img_dir=img_dir)
        train_df , val_df = train_test_split(df, test_size=train_ratio, random_state=randoms_state)

    print('num of train_df',len(train_df))
    print('num of val_df',len(val_df))

    num_cls = len(train_df.columns) - 1  # because, it is multi-label

    print('number of class: ', num_cls)
    cls_list = list(train_df.columns)
    cls_list.remove('id')
    print(cls_list)
    
    return train_df, val_df, num_cls, cls_list


class CustomDataset(Dataset):

    def __init__(self, dataframe, image_dir, num_classes, class_list, transforms=None, img_resize = False, img_dsize = (640,640)):
        super().__init__()
        
        self.image_ids = dataframe['id'].unique() # 이미지 고유 ID
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.img_resize = img_resize
        self.img_dsize = img_dsize
        self.class_list = class_list
        self.num_classes = num_classes
    

    def __getitem__(self, index: int):
        # 이미지 index로 아이템 불러오기

        image_id = self.image_ids[index]
        records = self.df[self.df['id'] == image_id]
        
        image = cv2.imread(f'{self.image_dir}/{image_id}', cv2.IMREAD_COLOR)
            
        # OpenCV가 컬러를 저장하는 방식인 BGR을 RGB로 변환
        if self.img_resize:
            image = cv2.resize(image, self.img_dsize)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        image /= 255.0 # 0 ~ 1로 스케일링

        target = np.array(records[self.class_list].values).astype(np.float32)
        target = target.reshape(-1)

        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]