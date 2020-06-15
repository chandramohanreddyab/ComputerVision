# Importing the basis libraries
import numpy as np
import pandas as pd

# Importing the cv and Keras for model training
import cv2
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator
from keras.layers import Dense, Convolution2D, Dropout, MaxPool2D, BatchNormalization, Flatten
from keras.optimizers import adam, RMSprop
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
import tensorflow as tf
from keras.utils import to_categorical

# Loading the data from Kaggle facial emotion recognization competition 2013
# (https://github.com/atulapra/Emotion-detection/blob/master/Tensorflow/emotions.py)
# Data has three columns (Emotion Label,Usage whether training or test,pixel values)

train_data = pd.read_csv('data/train.csv')
y = train_data['emotion']
X = [i.split(' ') for i in train_data['pixels']]  # fixel values are seperated by space

# -------------------------------------------------------------
num_classes = 7  # 0 to 6 as angry, disgust, fear, happy, sad and surprise, neutral
batch_size = 256
epochs = 30
# -------------------------------------------------------------
x_train = np.array(X, 'float32')  # X is list, so converted to array
y_train = y.astype('float32')
x_train /= 255  # normalizing pixel values such that values would be between 0 and 1

y_train = to_categorical(y_train, num_classes)  # conveting categorical varible
x_train = x_train.reshape(-1, 48, 48, 1)

model = Sequential()
# layer1
model.add(Convolution2D(32, (4, 4), padding = 'same', input_shape = (48, 48, 1)))
model.add(Convolution2D(64, (4, 4), padding = 'same'))
model.add(MaxPool2D())
model.add(Dropout(0.3))
# layer2
model.add(Convolution2D(64, (4, 4), padding = 'same'))
model.add(MaxPool2D())
model.add(Dropout(0.3))
# layer3
model.add(Convolution2D(128, (4, 4), padding = 'same'))
model.add(MaxPool2D())
model.add(Convolution2D(128, (4, 4), padding = 'same'))
model.add(MaxPool2D())
model.add(Dropout(0.3))
model.add(Flatten())

# fully connected neural networks
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(1024, activation = 'relu'))
model.add(Dropout(0.2))
model.add(Dense(num_classes, activation = 'softmax'))

# Model
# --------------------------------------------------------------
def emotion_detection(type=None, model=None):
    if type == 'train':
        # ------------------------------
        # Image augmentation and training with batch process
        ada = adam()
        gen = ImageDataGenerator()
        train_generator = gen.flow(x_train, y_train, batch_size = batch_size)
        model.compile(loss = 'categorical_crossentropy', optimizer = ada, metrics = ['accuracy'])
        # ------------------------------
        model.fit_generator(train_generator, steps_per_epoch = batch_size,
                            epochs = epochs)  # train for randomly selected one
        model.save_weights('data/facial_expression_detection_model.h5')
    else:
        model.load_weights('data/facial_expression_detection_model.h5')
    return model
