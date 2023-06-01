import os
import cv2
import torchvision.transforms as transforms

from xeception_2x1ch import *
from xeception import *
from googlenetv4 import *
import numpy as np

def check_path(path):
    if isinstance(path, list):
        if all(isinstance(item, np.ndarray) for item in path):
            print("List of OpenCV images received!")
            return 'list-of-images'
        else:
            print("List received but contains non-OpenCV images.")
            return None
        
    if os.path.isfile(path):
        if path.endswith(('.jpg', '.jpeg', '.png', '.bmp')):
            print("single image_classification!")
            return 'single-image'
        else:
            print(f"{path} is a file but not an image file.")
            return None
    elif os.path.isdir(path):
        print(f"{path} is a directory.")
        return 'directory-path'
    
    else:
        print(f"{path} does not exist.")
        return None

def load_image(image_path,img_size):
    transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(img_size)
    ])
    x_img = cv2.cvtColor(cv2.resize(cv2.imread(image_path),(640,640)),cv2.COLOR_BGR2RGB).astype(np.float32)
    x_img /= 255.0 # 0 ~ 1로 스케일링
    x_img = transformation(x_img)

    return x_img

def transfor_image(image,img_size):
    transformation = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Resize(img_size)
    ])
    x_img = cv2.cvtColor(cv2.resize(image,(640,640)),cv2.COLOR_BGR2RGB).astype(np.float32)
    x_img /= 255.0 # 0 ~ 1로 스케일링
    x_img = transformation(x_img)

    return x_img

#디렉토리로 올경우 배치처리로 ㄱㄱ
def process_images_in_batches(image_paths,img_size, batch_size, device, is_NOT_loaded=True):
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i : i + batch_size]
        if is_NOT_loaded:
            batch_images = [load_image(image_path,img_size) for image_path in batch_paths]
        else:
            batch_images = [transfor_image(image_path,img_size) for image_path in batch_paths]
        batch_tensor = torch.stack(batch_images).to(device)
        yield batch_tensor

def classfication(img_path, selected_model='googlenetv4', img_size=640, weight_path =None ):
    if selected_model == 'xeception':
        if weight_path == None:
            weight_path = './runs/pre_model/weight.pt'
        model = Xception(num_classes=7)
        model.load_state_dict(torch.load(weight_path))
    elif selected_model == 'googlenetv4':
        if weight_path == None:
            weight_path = "./runs/pre_model/weight.pt"
        model = InceptionV4(num_classes=7)
        model.load_state_dict(torch.load(weight_path))
    elif selected_model == 'VIT':
        if weight_path == 'None':
            weight_path = "^^"
        pass
    
    #gpu로가자
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    mymodel = model.to(device)
    
    #이미지 체크 리스트 작성
    check_list = []
    is_NOT_loaded = True
    inputed_datatype = check_path(img_path)
    if inputed_datatype == 'single-image':
        check_list.append(img_path)
    elif inputed_datatype == 'directory-path':
        check_list = [os.path.join(img_path, f) for f in os.listdir(img_path) if f.endswith(('.jpg', '.jpeg', '.png', '.bmp'))]
    elif inputed_datatype == 'list-of-images':
        check_list = img_path
        is_NOT_loaded = False
    else:
        return None

    result_list = []
    for batch in process_images_in_batches(check_list, img_size, 2, device, is_NOT_loaded):
        mymodel.eval() # 모델을 평가 모드로 설정
        with torch.no_grad():
            output = mymodel(batch)
            pred = output.sigmoid() >= 0.5
            pred_list = pred.cpu().numpy().tolist()
            result_list.extend(pred_list)
    return check_list, result_list