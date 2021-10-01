import keras
from keras.datasets import mnist 
from keras import regularizers
from keras.models import Sequential , Model
from keras.layers import Dense, Dropout, Flatten ,Input
from keras.layers import Conv2D, MaxPooling2D , BatchNormalization
from keras import backend as K 
from keras.models import load_model
from keras.utils import multi_gpu_model
from keras.callbacks import Callback
from keras.callbacks import LearningRateScheduler
from data_processing import get_last_checkpoint
from keras.utils import multi_gpu_model

class ChangCnn : 
    def __init__(self, classes, learning_rate, gpu_config) : 
        self.model = Sequential()
        self.model.add(Conv2D(64, kernel_size=(5,5),padding = 'same', activation='relu',input_shape=(256,256,3), strides=(2,2), use_bias = False))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(64,(5,5),padding='same', activation='relu', strides=(2,2), use_bias = False))
        self.model.add(BatchNormalization())
        self.model.add(Conv2D(32,(3,3),padding='same', activation='relu', strides=(1,1), use_bias = False))
        self.model.add(BatchNormalization())
        self.model.add(MaxPooling2D(pool_size=(3,3), strides=(2,2), padding='valid'))
        self.model.add(Dropout(0.2))
        self.model.add(Flatten())
        self.model.add(Dense(256, activation='relu', use_bias = False))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(4096, activation='relu', use_bias = False))
        self.model.add(BatchNormalization())
        self.model.add(Dropout(0.5))
        self.model.add(Dense(classes, activation='softmax'))
        self.model.summary()

        if gpu_config["multi_gpu"] : 
            self.model = multi_gpu_model(self.model, gpu_config["n_gpus"])

        self.model.compile(loss=keras.losses.categorical_crossentropy, 
                           optimizer=keras.optimizers.Adam(lr=learning_rate), 
                           metrics=['accuracy'])
