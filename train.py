import data_processing
from ChangCnn import ChangCnn
import utils

import os
from glob import glob
from PIL import Image
import numpy as np
import cv2
import matplotlib.pyplot as plt
import random
import tensorflow as tf
import keras
import datetime
import csv
import multiprocessing as mp
from sklearn.metrics import confusion_matrix
from tensorflow.python.keras.callbacks import TensorBoard
from Datagenerator import ImageDataGenerator
from sklearn.preprocessing import OneHotEncoder 
from sklearn.preprocessing import LabelEncoder
import json
from sklearn.model_selection import train_test_split
from itertools import groupby

"""
data = np.load(EXIF_NPY_PATH).item()  # 각 이미지의 exif BrightnessValue 정보를 담은 npy file read
X_train = data["X_train"]; y_train = data["y_train"]
X_val = data["X_val"]; y_val = data["y_val"]
X_test = data["X_test"]; y_test = data["y_test"]

print("Train", set(y_train)) # Train set의 중복되지 않은 label 출력
print("Validation", set(y_val))   # Val set의 중복되지 않은 label 출력
print("Test", set(y_test))  # Test set의 중복되지 않은 label 출력

assert len(set(y_val) - set(y_train)) == 0 # validation의 label에 train에 없는 label이 있는 경우 
assert len(set(y_test) - set(y_train)) == 0 # test의 label에 train에 없는 label이 있는 경우 

n_classes = len(set(y_train))    # 클래스 개수 = train set의 중복되지 않은 label 개수
"""

# Error when checking target 발생하는 경우
# 1. one hot encoding 확인 (https://stackoverflow.com/questions/46177721/keras-error-when-checking-target/46178460)
# 2. numpy array인지 확인 (https://stackoverflow.com/questions/42596057/keras-error-expected-to-see-1-array)


if __name__ == "__main__" :
    CONFIG_PATH = 'C:/Users/rlfalsgh95/source/repos/Camera_dataset_experiment/Camera_dataset_experiment/identification_experiment/5/config.json'

    with open(CONFIG_PATH, "r", encoding= "utf8") as config:
        data = config.read()
        tc = json.loads(data)["train"]
        dc = json.loads(data, object_hook= utils.hinted_tuple_hook)["data"]

    ###cudnn error
    tensorflowConfig = tf.ConfigProto()
    tensorflowConfig.gpu_options.allow_growth = True
    session = tf.Session(config=tensorflowConfig)

    os.environ['CUDA_VISIBLE_DEVICES']= str(tc["gpu_num"]) # set gpu num

    EXPERIMENT_PATH = os.path.dirname(CONFIG_PATH)
    PICKLE_PATH = os.path.join(EXPERIMENT_PATH, "pickle")
    LOG_PATH = os.path.join(EXPERIMENT_PATH, "logs")

    TRAIN_PICKLE_PATH = os.path.join(PICKLE_PATH, "train")
    VAL_PICKLE_PATH = os.path.join(PICKLE_PATH, "val")
    TEST_PICKLE_PATH = os.path.join(PICKLE_PATH, "test") 

    CHECKPOINT_DIR_PATH = os.path.join(EXPERIMENT_PATH, "Checkpoint")   # Directory path to store checkpoints
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR_PATH, "{epoch:08d}.h5") # Path of Checkpoint file 

    # Create ModelCheckPoint, Tensorboard Object
    now = datetime.datetime.now().strftime('%Y.%m.%d %H-%M-%S') # Current time
    model_checkpoint = keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, monitor = "accuracy", period = 1)   
    tensorboard = TensorBoard(log_dir = "{}/{}".format(LOG_PATH, now), update_freq = tc["batch_size"] * tc["log_interval"])  


    # Load train, test datasets path 
    train_max = tc["train_max"] if "train_max" in tc else None
    X, y = data_processing.get_datasets_path(dc["train_path"], desc = "train", max_per_class = train_max)

    # Split the dataset into train/val/test.
    if "test_path" not in dc : 
        if "train_rate" not in tc or "val_rate" not in tc or "test_rate" not in tc : 
            raise Exception("If the test path is not specified, train_rate, val_rate, and test_rate must all be specified in config file.")

        origin_X_train, origin_X_test, origin_y_train, origin_y_test = train_test_split(X, y, test_size = 1 - tc["train_rate"], random_state = tc["seed"]) 
        origin_X_val, origin_X_test, origin_y_val, origin_y_test= train_test_split(origin_X_test, origin_y_test, test_size = tc["test_rate"] / (tc["test_rate"] + tc["val_rate"]) , random_state = tc["seed"])
    else : 
        if "train_rate" in tc or "test_rate" in tc : 
            raise Exception("If test_path is specified, train_rate and test_rate are not required.")

        origin_X_train, origin_X_val, origin_y_train, origin_y_val = train_test_split(X, y, test_size = tc["val_rate"], random_state = tc["seed"])
        origin_X_test, origin_y_test = data_processing.get_datasets_path(dc["test_path"], desc = "test")

    print("# of train/val/test : ", len(origin_y_train), len(origin_y_val), len(origin_y_test))

    # Check that the # of unique labels in each train, val, and test is the same as the # of classes.
    assert tc["classes"] == len(np.unique(origin_y_train)), "# of train labels is different from the # of classes."
    assert tc["classes"] == len(np.unique(origin_y_val)), "# of val labels is different from the # of classes."
    assert tc["classes"] == len(np.unique(origin_y_test)), "# of test labels is different from the # of classes."

    # Split train/val/test images into patches and convert them to numpy arrays
    load_img_option = dc["load_img_option"]

    X_train, y_train = data_processing.img_to_array(origin_X_train, origin_y_train, TRAIN_PICKLE_PATH, load_img_option)   
    X_val, y_val = data_processing.img_to_array(origin_X_val, origin_y_val, VAL_PICKLE_PATH, load_img_option) 
    X_test, y_test = data_processing.img_to_array(origin_X_test, origin_y_test, TEST_PICKLE_PATH, load_img_option) 

    # Train/val label OneHotEncoding
    encoder, y_train = data_processing.encode_label(y_train, True)  
    y_val = encoder.transform(np.array(y_val).reshape(-1, 1)).toarray() 
    y_test = encoder.transform(np.array(y_test).reshape(-1, 1)).toarray() 

    if not os.path.exists(CHECKPOINT_DIR_PATH) : # If the directory to store the checkpoint does not exist, create it
        os.makedirs(CHECKPOINT_DIR_PATH)

    model = ChangCnn(tc["classes"], tc["learning_rate"]).model
    model.fit(np.array(X_train), np.array(y_train), validation_data = (np.array(X_val), np.array(y_val)),
              epochs = tc["epochs"], batch_size = tc["batch_size"], callbacks = [tensorboard, model_checkpoint])  # 모델 학습

