import os
from glob import glob
from PIL import Image
import numpy as np
import cv2
import random
import tensorflow as tf
import keras
from keras import backend as K 
from patch_extractor import patch_extractor_one_arg
from collections import OrderedDict
from data_processing import encode_label, get_img_size
from PIL import Image
import copy
from tqdm import tqdm
from pprint import pprint
from multiprocessing import cpu_count, Pool
from itertools import product

class ImageDataGenerator(keras.utils.Sequence):
    def __init__(self, 
                 image_paths, 
                 labels, 
                 pickle_dir_path,
                 desc = None,
                 find_label = None,
                 label_list = None,
                 one_hot = False,
                 encoder = None,
                 patch_option = None, 
                 batch_size = 128, 
                 n_channels = 3, 
                 scaling = True) :
        
        self.image_paths = image_paths
        self.labels = labels
        self.pickle_dir_path = pickle_dir_path
        self.desc = desc
        self.find_label = find_label
        self.label_list = label_list
        self.one_hot = one_hot
        self.encoder = encoder
        self.patch_option = patch_option
        self.batch_size = batch_size
        self.n_channels = n_channels
        self.scaling = scaling

        
        if self.one_hot and self.encoder is not None : # one_hot과 encoder parameter 모두 지정한 경우
            self.labels = self.encoder.transform(np.array(self.labels).reshape(-1, 1)).toarray() 
        elif self.one_hot and self.encoder is None : # one_hot은 지정하였지만, encoder는 지정하지 않은 경우
            self.encoder, self.labels = encode_label(self.labels, True)
        elif not self.one_hot and self.label_list is None : # one_hot과 label_list parameter 모두 지정하지 않은 경우
            raise Exception("You must specify label_list or one_hot parameter.")
        elif not self.one_hot and self.label_list is not None and self.find_label is None : # one_hot과 find_label parameter를 지정해주지 않고, label_list만 지정한 경우 find_label을 지정하도록 함.
            raise Exception("If label list is specified, the find_label function must be specified.")

        self.batch_dir_path = os.path.join(self.pickle_dir_path, "batch") # 각 batch 파일을 저장할 dir의 path

        # pickle dir과 batch dir이 없는 경우 생성
        if not os.path.exists(self.pickle_dir_path) : 
            os.makedirs(self.pickle_dir_path)

        if not os.path.exists(self.batch_dir_path) : 
            os.makedirs(self.batch_dir_path)

        all_patch_path , y = [], []

        if self.patch_option is not None : # patch option을 지정한 경우
            patch_idx = 0
            pool = Pool(int(cpu_count()))

            patch_info_path = os.path.join(self.pickle_dir_path, "all_patch_path.npy")  # 모든 patch에 대한 정보를 저장할 경로

            if not os.path.exists(patch_info_path) : 
                for idx, image_path in enumerate(tqdm(self.image_paths, desc = self.desc + " save patch batch")) : # 각 이미지의 patch를 추출하고, 각 patch를 npy 형태로 저장
                    with Image.open(image_path) as img : # 이미지 불러옴
                        patch_option = copy.copy(self.patch_option) # patcg_option의 값을 수정하지 않기 위해 shallow copy
                        array_img = np.asarray(img) # img to numpy array

                        patch_option["img"] = array_img # patch option에 image array 추가
                        patches = patch_extractor_one_arg(patch_option) # patch_option에 따라 이미지로부터 patch 추출
                        n_patches = len(patches) # 추출된 patch의 개수
                    
                        patch_paths = [os.path.join(self.pickle_dir_path, "patch_" + str(i) + ".npy") for i in range(patch_idx, patch_idx + n_patches)] # 각 patch의 저장 위치
                        pool.starmap(np.save, zip(patch_paths, patches)) # 각 patch를 저장
                    
                        patch_idx += n_patches 

                        all_patch_path += patch_paths # all_patch_path에 각 patch 파일의 path를 저장
                        y += [self.labels[idx]] * n_patches # 각 patch의 label 저장

                np.save(patch_info_path, {"all_patch_path" : all_patch_path, "y" : y}) # 모든 패치 정보 저장
            else :
                patch_info = np.load(patch_info_path).item()
                all_patch_path = patch_info["all_patch_path"]
                y = patch_info["y"]

            # all_patch_path와 label shuffle
            indices = np.arange(len(all_patch_path))
            np.random.shuffle(indices)
            all_patch_path = np.array(all_patch_path)[indices]
            y= np.array(y)[indices]

            patch_idx = 0
            batch_idx = 0

            while True : 
                if patch_idx + self.batch_size > len(all_patch_path) : # batch_size로 나누어 떨어지지 않는 patch는 사용하지 않음.
                    break;

                batch_patch_paths = all_patch_path[patch_idx : patch_idx + self.batch_size] # 전체 patch에서 batch size만큼 떼어냄
                batch_label = y[patch_idx : patch_idx + self.batch_size]

                patch_idx += self.batch_size
                
                batch_path = os.path.join(self.batch_dir_path, "batch_" + str(batch_idx) + ".npy")   # batch file의 path
                batch_idx += 1

                if os.path.exists(batch_path) : # 이미 존재하는 경우 continue
                    continue

                batch = pool.map(np.load, batch_patch_paths) # batch를 구성할 각 patch를 load함.

                np.save(batch_path, {"x" : batch, "y" : batch_label}) # batch_size만큼의 patch들을 묶어 하나의 batch 파일로 저장
        
            pool.close()

            self.n_images = len(all_patch_path)
        else : 
            self.n_images = len(self.image_paths)
        
        if desc is not None : 
            print(self.desc, "Total images : ", self.n_images)
        else : 
            print("Total images : ", self.n_images)

        self.on_epoch_end()

    def __len__(self):
        return int(np.floor(self.n_images / self.batch_size))


    def __getitem__(self, index):
        batch = np.load(os.path.join(self.batch_dir_path, "batch_" + str(index) + ".npy")).item()
        X = np.array(batch["x"])
        y = np.array(batch["y"])

        if self.scaling : 
            X = X / 255
        
        return X, y
