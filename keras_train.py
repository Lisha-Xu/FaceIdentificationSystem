# -*- coding: utf-8 -*-
import numpy as np
from numpy import *
from time import time
import keras
from sklearn.model_selection import train_test_split

from keras.models import Sequential
from keras.layers.normalization import BatchNormalization
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.layers.core import Activation
from keras.layers.core import Flatten
from keras.layers.core import Dropout
from keras.layers.core import Dense
from keras import backend as K
from sklearn.preprocessing import LabelBinarizer
from keras.optimizers import Adam
from keras.callbacks import ModelCheckpoint
from keras.models import load_model
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
import pickle
import argparse
import os
from keras.callbacks import TensorBoard
from keras.utils import plot_model
import matplotlib.pyplot as plt
EPOCHS = 120
INIT_LR = 1e-3
BS = 36


class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(32, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(512))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model

'''
class SmallerVGGNet:
    @staticmethod
    def build(width, height, depth, classes):
        model = Sequential()
        inputShape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            inputShape = (depth, height, width)
            chanDim = 1

        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Conv2D(64, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(3, 3)))
        model.add(Dropout(0.25))
        model.add(Conv2D(128, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(128, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Conv2D(512, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(Conv2D(512, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(Conv2D(512, (3, 3), padding="same", input_shape=inputShape))
        model.add(Activation("relu"))
        model.add(BatchNormalization(axis=chanDim))
        model.add(MaxPooling2D(pool_size=(2, 2)))
        model.add(Dropout(0.25))
        model.add(Flatten())
        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(Dense(4096))
        model.add(Activation("relu"))
        model.add(BatchNormalization())
        model.add(Dropout(0.5))

        model.add(Dense(classes))
        model.add(Activation("softmax"))
        return model
'''

def trainKeras():
    data = np.load("app/data_keras.npy")
    label = np.load("app/label_keras.npy")
    data = data / 255
    classes = len(np.unique(label))
    for i in range(1, 10000):
        x_train, x_val, y_train, y_val = train_test_split(
            data, label, test_size=0.25, random_state=i)
        if len(np.unique(y_train)) == len(np.unique(y_val)):
            break

    lb = LabelBinarizer()
    lb.fit(label)
    y_train = lb.fit_transform(y_train)
    y_val = lb.fit_transform(y_val)

    model = SmallerVGGNet.build(96, 96, 3, classes=classes)
    opt = Adam(lr=INIT_LR, decay=INIT_LR / EPOCHS)
    model.compile(loss="categorical_crossentropy", optimizer=opt, metrics=["accuracy"])
    '''
    #history = LossHistory()
    #t0 = time()
    train_log = model.fit(x_train, y_train, batch_size=BS, validation_data=(x_val, y_val),
                  epochs=EPOCHS, verbose=1,callbacks=[history])
    #t = time() - t0
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    #history.loss_plot('epoch')
    model.save("app/face.h5")
    return score[1]
    '''
    tensorboard = TensorBoard(log_dir='log')
    filepath = "app/face.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_acc', save_best_only='True', save_weights_only=False,
                                 mode='max', period=1)
    callback_lists = [tensorboard, checkpoint]
    t0 = time()
    train_log = model.fit(x_train, y_train, batch_size=BS, validation_data=(x_val, y_val),
                          epochs=EPOCHS, verbose=1, callbacks=callback_lists)
    x = time()-t0
    model = load_model("app/face.h5")
    score = model.evaluate(x_val, y_val, verbose=0)
    print('Test score:', score[0])
    print('Test accuracy:', score[1])
    return score[1],x
