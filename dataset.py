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
    val_csv_path = None if val_csv_path == 'None' else val_csv_path
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


#이건 과제용임!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def get_data_from_csv_TH(csv_path, train_ratio, img_dir, randoms_state=42):
    ###### columns example : ['id', 'good', 'b_edge', 'burr', 'borken', 'b_bubble', 'etc', 'no_lens']
    print('####################################################################################')
    print('############################# 과제용임!!!!!!!! ######################################')
    print('####################################################################################')


    df = pd.read_csv(csv_path)
    df = validate_dataset(df=df,img_dir=img_dir)
    train_df , temp_df = train_test_split(df, test_size=1-train_ratio, random_state=randoms_state)
    val_df, test_df = train_test_split(temp_df, test_size=0.5, random_state=randoms_state)


    # 'good' 열을 데이터 프레임에서 제거 및 클래스 재정렬
    cls_list = ['no_lens', 'etc', 'burr', 'borken', 'b_edge', 'b_bubble']
    train_df = train_df.drop(columns=['good'])
    train_df = train_df[['id'] + cls_list]
    val_df = val_df.drop(columns=['good'])
    val_df = val_df[['id'] + cls_list]
    test_df = test_df.drop(columns=['good'])
    test_df = test_df[['id'] + cls_list]


    print('num of train_df',len(train_df))
    print('num of val_df',len(val_df))
    print('num of test_df',len(test_df))

    num_cls = len(train_df.columns) - 1  # because, it is multi-label

    print('number of class: ', num_cls)
    # cls_list = list(train_df.columns)
    # cls_list.remove('id')

    print(cls_list)

    return train_df, val_df, test_df, num_cls, cls_list


class CustomDataset(Dataset):

    def __init__(self, dataframe, image_dir, num_classes, class_list, transforms=None, img_resize = False, img_dsize = (640,640), polar_tranform = False):
        super().__init__()
        
        self.image_ids = dataframe['id'].unique() # 이미지 고유 ID
        self.df = dataframe
        self.image_dir = image_dir
        self.transforms = transforms
        self.img_resize = img_resize
        self.img_dsize = img_dsize
        self.class_list = class_list
        self.num_classes = num_classes
        self.polar_transform = polar_tranform

    #데이터 길이 검증
    def validate_data_records(self):
        for idx, image_id in enumerate(self.image_ids):
            records = self.df[self.df['id'] == image_id]
            target = np.array(records[self.class_list].values).astype(np.float32)
            if target.shape[1] != len(self.class_list):
                print(f"Index {idx} with image_id {image_id} has mismatched target size. Expected {len(self.class_list)}, but got {target.shape[1]}")

    def polar_transform_func(self, pic):
        pic = cv2.linearPolar(pic, (pic.shape[0]/2,pic.shape[1]/2), pic.shape[0]/2, cv2.WARP_FILL_OUTLIERS)
        pic = cv2.rotate(pic, cv2.ROTATE_90_COUNTERCLOCKWISE)
        img_edge = pic[:160,:,:]
        
        return img_edge

    def __getitem__(self, index: int):
        # 이미지 index로 아이템 불러오기

        image_id = self.image_ids[index]
        records = self.df[self.df['id'] == image_id]
        
        image = cv2.imread(f'{self.image_dir}/{image_id}', cv2.IMREAD_COLOR)
            
        # OpenCV가 컬러를 저장하는 방식인 BGR을 RGB로 변환
        if self.img_resize and not self.polar_transform:
            image = cv2.resize(image, self.img_dsize)

        if self.polar_transform:
            image = self.polar_transform_func(image)
            if self.img_resize:
                new_width = self.img_dsize[0]
                new_height = int(image.shape[0] * (new_width / image.shape[1]))
                image = cv2.resize(image, (new_width, new_height))
                

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB).astype(np.float32)
        #cv2.imwrite( 'sample/img.jpg', image)
        image /= 255.0 # 0 ~ 1로 스케일링

        target = np.array(records[self.class_list].values).astype(np.float32)
        target = target.reshape(-1)
        if self.transforms is not None:
            image = self.transforms(image)

        return image, target

    def __len__(self) -> int:
        return self.image_ids.shape[0]