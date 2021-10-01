import data_processing
from ChangCnn import ChangCnn
import utils
from ImageDatagenerator import ImageDataGenerator

import os
import numpy as np
import json
import datetime
from pprint import pprint
from tqdm import tqdm
import matplotlib.pyplot as plt
import itertools
import io

import keras
import tensorflow as tf
from tensorflow.python.keras.callbacks import TensorBoard
from keras.models import load_model

import sklearn
import multiprocessing as mp
from multiprocessing import cpu_count
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split

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


def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.
    
    Args:
       cm (array, shape = [n, n]): a confusion matrix of integer classes
       class_names (array, shape = [n]): String names of the integer classes
    """
    
    figure = plt.figure(figsize=(20, 20))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=45)
    plt.yticks(tick_marks, class_names)
    
    # Normalize the confusion matrix.
    cm = np.around(cm.astype('float') / cm.sum(axis=1)[:, np.newaxis], decimals=2)
    
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.
    
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, cm[i, j], horizontalalignment="center", color=color)
        
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    return figure

def plot_to_image(figure):
    """
    Converts the matplotlib plot specified by 'figure' to a PNG image and
    returns it. The supplied figure is closed and inaccessible after this call.
    """
    
    buf = io.BytesIO()
    
    # Use plt.savefig to save the plot to a PNG in memory.
    plt.savefig(buf, format='png')
    
    # Closing the figure prevents it from being displayed directly inside
    # the notebook.
    plt.close(figure)
    buf.seek(0)
    
    # Use tf.image.decode_png to convert the PNG buffer
    # to a TF image. Make sure you use 4 channels.
    image = tf.image.decode_png(buf.getvalue(), channels=4)
    
    # Use tf.expand_dims to add the batch dimension
    image = tf.expand_dims(image, 0)
    
    return image

def log_confusion_matrix(y_true, y_pred, labels, title, log_dir, epoch):
    # Calculate the confusion matrix using sklearn.metrics
    cm = sklearn.metrics.confusion_matrix(y_true, y_pred)
    
    figure = plot_confusion_matrix(cm, class_names=labels)
    cm_image = plot_to_image(figure)
    

    # Log the confusion matrix as an image summary.
    
    #with file_writer_cm.as_default():
    with tf.Session() as sess : 
        file_writer_cm = tf.summary.FileWriter(log_dir + '/Confusion_matrix', session = sess)

        img = tf.summary.image(title, cm_image).eval()
        file_writer_cm.add_summary(img, global_step=epoch)
        file_writer_cm.flush()

if __name__ == "__main__" :
    CONFIG_PATH = 'C:/Users/rlfalsgh95/source/repos/Camera_dataset_experiment/identification_experiment/6/config.json'
    
    with open(CONFIG_PATH, "r", encoding= "utf8") as config:
        data = config.read()
        tc = json.loads(data, object_hook= utils.hinted_tuple_hook)["train"]
        dc = json.loads(data, object_hook= utils.hinted_tuple_hook)["data"]
    gpu_config = tc["gpu"]

    ###cudnn error
    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config)
    
    EXPERIMENT_PATH = os.path.dirname(CONFIG_PATH)
    PICKLE_PATH = os.path.join(EXPERIMENT_PATH, "pickle")
    LOG_PATH = os.path.join(EXPERIMENT_PATH, "logs")

    TRAIN_PICKLE_PATH = os.path.join(PICKLE_PATH, "train")
    VAL_PICKLE_PATH = os.path.join(PICKLE_PATH, "val")
    TEST_PICKLE_PATH = os.path.join(PICKLE_PATH, "test") 
    TEST_RESULT_PATH = os.path.join(PICKLE_PATH, "test_result.npy")

    CHECKPOINT_DIR_PATH = os.path.join(EXPERIMENT_PATH, "Checkpoint")   # Directory path to store checkpoints
    CHECKPOINT_PATH = os.path.join(CHECKPOINT_DIR_PATH, "{epoch:08d}.h5") # Path of Checkpoint file 

    # Load train, test datasets path 
    train_max = tc["train_max"] if "train_max" in tc else None

    data_processing.get_datasets_path(dc["train_path"], desc = "Train") # print num of image per class (max_per_class = None)
    X, y = data_processing.get_datasets_path(dc["train_path"], desc = "Train", max_per_class = train_max)

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
        test_max = tc["test_max"] if "test_max" in tc else None
        
        data_processing.get_datasets_path(dc["test_path"], desc = "Test") # print num of image per class (max_per_class = None)
        origin_X_test, origin_y_test = data_processing.get_datasets_path(dc["test_path"], desc = "Test", max_per_class = test_max)

    pprint(data_processing.label_distribution(origin_y_train, "Train data After split"))
    pprint(data_processing.label_distribution(origin_y_val, "Val data After split"))
    pprint(data_processing.label_distribution(origin_y_test, "Test data After split"))

    print("# of train/val/test : ", len(origin_y_train), len(origin_y_val), len(origin_y_test))
    
    params = tc["generator"]

    # Check that the # of unique labels in each train, val, and test is the same as the # of classes.
    assert tc["n_classes"] == len(np.unique(origin_y_train)), "# of train labels is different from the # of classes."
    assert tc["n_classes"] == len(np.unique(origin_y_val)), "# of val labels is different from the # of classes."
    assert tc["n_classes"] == len(np.unique(origin_y_test)), "# of test labels is different from the # of classes."
    assert len(set(origin_y_train).difference(set(origin_y_test))) == 0

    if not os.path.exists(CHECKPOINT_DIR_PATH) : # If the directory to store the checkpoint does not exist, create it
        os.makedirs(CHECKPOINT_DIR_PATH)
       
 
    if gpu_config["multi_gpu"] and "gpu_num" in gpu_config: 
        raise Exception("You cannot specify a and b at the same time.")
    elif not gpu_config["multi_gpu"] : 
        os.environ['CUDA_VISIBLE_DEVICES']= str(gpu_config["gpu_num"]) # set gpu num

    if "load_checkpoint" in tc and tc["load_checkpoint"]: 
        checkpoint_path = data_processing.get_last_checkpoint(CHECKPOINT_DIR_PATH)
        initial_epoch = int(os.path.basename(os.path.splitext(checkpoint_path)[0])) + 1

        model = load_model(checkpoint_path)
    else :  
        initial_epoch = 0
        model = ChangCnn(tc["n_classes"], tc["learning_rate"], gpu_config).model
   
    print(origin_X_test)

    if tc["workers"] == "max" : 
        workers = cpu_count()
    else : 
        workers = int(tc["workers"])

    # Create ModelCheckPoint, Tensorboard Object
    now = datetime.datetime.now().strftime('%Y.%m.%d %H-%M-%S') # Current time
    model_checkpoint = keras.callbacks.ModelCheckpoint(CHECKPOINT_PATH, monitor = "accuracy", period = 1)   
    tensorboard_update_freq = params["batch_size"] * tc["log_interval"]
    tensorboard = TensorBoard(log_dir = "{}/{}".format(LOG_PATH, now), update_freq = tensorboard_update_freq, write_graph= False)  

    training_generator = ImageDataGenerator(origin_X_train, origin_y_train, desc = "Train", pickle_dir_path = TRAIN_PICKLE_PATH, **params)
    validation_generator = ImageDataGenerator(origin_X_val, origin_y_val, encoder = training_generator.encoder, pickle_dir_path = VAL_PICKLE_PATH, desc = "Validation", **params)
    test_generator = ImageDataGenerator(origin_X_test, origin_y_test, encoder = training_generator.encoder, pickle_dir_path = TEST_PICKLE_PATH, desc = "test", **params)
    3 / 0
    
    #model.fit_generator(generator=training_generator, validation_data=validation_generator, use_multiprocessing=False,
     #         epochs = tc["n_epochs"], workers=workers, callbacks = [tensorboard, model_checkpoint], initial_epoch = initial_epoch)  # 모델 학습
    
    y_trues = []
    y_preds = []

    
    print("Predict...")
    if not os.path.exists(TEST_RESULT_PATH) : 
        #Confution Matrix and Classification Report
        for batch, y in tqdm(test_generator) : 
            y_pred = model.predict(batch)
            y_preds += list(y_pred)
            y_trues += list(y)

        np.save(TEST_RESULT_PATH, {"y_pred" : y_preds, "y_true" : y_trues})
    else : 
        test_result = np.load(TEST_RESULT_PATH).item()
        y_preds = test_result["y_pred"]
        y_trues = test_result["y_true"]
    
    print("Calculate accuracy...")
    eval_result = model.evaluate_generator(test_generator, verbose = 1)
    loss  = eval_result[0]
    accuracy = eval_result[1]

    print("loss: %.3f - acc: %.3f" % (loss, accuracy))

    print("Confusion matrix...")
    log_confusion_matrix(np.argmax(y_trues, axis = 1), np.argmax(y_preds, axis=1), 
                         labels = training_generator.encoder.categories_[0], title = "confusion_matrix (loss: %.3f - acc: %.3f)" % (loss, accuracy), log_dir = LOG_PATH, epoch =1)











    