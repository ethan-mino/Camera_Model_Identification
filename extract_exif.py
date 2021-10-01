import os
from tqdm import tqdm
from pprint import pprint
from PIL.ExifTags import TAGS
from PIL import Image
from sklearn.model_selection import train_test_split
from collections import OrderedDict
import numpy as np
from functools import reduce

def get_exif(image_path) : 
    img = Image.open(image_path)
    exif_data = img._getexif()
    exif = {TAGS[k] : v for k, v, in exif_data.items() if k in TAGS}
    
    return exif


def get_files_path(dir_path) : 
    f = lambda x, y, z : list(map(lambda filename : os.path.join(x, filename), z))  # 파일명과 path 연결
    image_path_list = [j for i in os.walk(dir_path) for j in f(*i)]    # 실제 소유중인 파일의 path 목록
    
    return image_path_list

if __name__ == "__main__" : 
    PROJECT_PATH = "C:/Users/rlfalsgh95/source/repos/Camera_dataset_experiment"
    SMDB_DIR = os.path.join(PROJECT_PATH, "data/original/SMDB")

    PHONE_DIR_NAME = "smartphone_photo"
    PHONE_IMAGE_DIR = os.path.join(SMDB_DIR, PHONE_DIR_NAME)

    EXIF_NPY_NAME = "ShutterSpeedValue.npy"
    EXIF_NPY_PATH = os.path.join(PROJECT_PATH, EXIF_NPY_NAME)

    attr_name= "ShutterSpeedValue"

    SEED = 49

    TRAIN_RATE = 0.64
    VALIDATION_RATE = 0.16
    TEST_RATE = 0.2 

    N_IMAGE = 200

    np.random.seed(SEED)    # Random Seed 설정

    camera_model_list = os.listdir(PHONE_IMAGE_DIR) # 카메라 모델명 list

    image_index = 0
    DATA_SPLIT_SEED = 50 # train, val, test를 분할하는데 사용하는 seed

    X_train = []
    y_train = []

    X_val = []
    y_val = []

    X_test= []
    y_test = []

    model_image_info = {}

    for camera_model in tqdm(camera_model_list, desc="extract exif"):
        camera_model_path = os.path.join(PHONE_IMAGE_DIR, camera_model)  # camera_model에 해당하는 카메라 모델의 이미지를 저장하는 dir의 path
        image_path_list = get_files_path(camera_model_path)  # camera_model에 해당하는 이미지들의 path

        np.random.shuffle(image_path_list)  # image_path shuffle
    
        img_index = 0
        X_model = []
        y_model = []

        for image_path in image_path_list : # shuffle된 image path에서 최대 N_IMAGE개만 사용
            exif_info = get_exif(image_path)

            try : # 해당 속성이 없는 이미지가 있어서 try catch
                attr_value = exif_info[attr_name]
                y_model.append(int(attr_value[0] / attr_value[1]))
                X_model.append(image_path)
                img_index += 1
            
                if img_index >= N_IMAGE : 
                    break;
            except Exception as err: 
                pass
    
        X_model_train = []; y_model_train = []
        X_model_test = []; y_model_test= []
        X_model_val = []; y_model_val = []

        if len(X_model) != 0 :     # 데이터의 수가 0이 아니라면
            try : 
                X_model_train, X_model_test, y_model_train, y_model_test = train_test_split(X_model, y_model, test_size = 1 - TRAIN_RATE, random_state = DATA_SPLIT_SEED) # 현재 model의 train, test 분할 
                X_model_val, X_model_test, y_model_val, y_model_test= train_test_split(X_model_test, y_model_test, test_size = TEST_RATE / (TEST_RATE + VALIDATION_RATE) , random_state = DATA_SPLIT_SEED) # 현재 모델의 test, val 분할
            except : 
                print(camera_model, len(X_model))

        X_train.extend(X_model_train), y_train.extend(y_model_train) # 전체 train 데이터에 현재 모델의 train 데이터 추가
        X_val.extend(X_model_val), y_val.extend(y_model_val)    # 전체 val 데이터에 현재 모델의 val 데이터 추가
        X_test.extend(X_model_test), y_test.extend(y_model_test) # 전체 test 데이터에 현재 모델의 test 데이터 추가
    
        model_image_info[camera_model] = OrderedDict({"total_image" : len(image_path_list), "extract_image" : len(X_model), "train" : len(X_model_train), "val" : len(X_model_val), "test" : len(X_model_test)}) #  현재 모델의 train, val, test 데이터 개수

    pprint(model_image_info)

    print("Total # of extract img : ", reduce (lambda x, value : x + value["extract_image"], model_image_info.values(), 0)) # 추출된 전체 이미지의 개수
    print("Total # of Train img : ", reduce (lambda x, value : x + value["train"], model_image_info.values(), 0))  # train 이미지의 개수
    print("Total # of Val img : ", reduce (lambda x, value : x + value["val"], model_image_info.values(), 0))  # val 이미지의 개수
    print("Total # of Test img : ", reduce (lambda x, value : x + value["test"], model_image_info.values(), 0)) # test 이미지의 개수

    np.save(EXIF_NPY_PATH, {"X_train" : X_train, "y_train" : y_train, "X_val" : X_val, "y_val" : y_val, "X_test" : X_test, "y_test" : y_test})