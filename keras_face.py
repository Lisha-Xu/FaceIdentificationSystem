# -*- coding: utf-8 -*-
# from skimage import io, transform
from datetime import datetime
import cv2
import glob
import os
import tensorflow as tf
import numpy as np
from numpy import *
import time
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
from keras.preprocessing.image import ImageDataGenerator
import pickle
import argparse
import os

EPOCHS = 100
INIT_LR = 1e-3
BS = 32

from app.dlib_test import read_im_and_landmarks, transformation_from_points, warp_im

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
PREDICTOR_PATH = "F:/biyesheji/shape_predictor_68_face_landmarks.dat"
FACE_POINTS = list(range(17, 68))
MOUTH_POINTS = list(range(48, 61))
RIGHT_BROW_POINTS = list(range(17, 22))
LEFT_BROW_POINTS = list(range(22, 27))
RIGHT_EYE_POINTS = list(range(36, 42))
LEFT_EYE_POINTS = list(range(42, 48))
NOSE_POINTS = list(range(27, 35))
JAW_POINTS = list(range(0, 17))

# Points used to line up the images.
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

im1, landmarks1 = read_im_and_landmarks("app/static/uploads/1.bmp")

path = "F:/biyesheji/PythonStudyCode/app/static/uploads/users_dup/"

# 将所有的图片resize成100*100

w = 128
h = 128
c = 3

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img

datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',
    cval=0)


def generate():

    dirs = os.listdir(path)
    for dir_name in dirs:
        subject_dir_path = path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        if os.path.isfile(subject_dir_path+"/1.txt"):
            continue;
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue;
            image_path = subject_dir_path + "/" + image_name
            img = load_img(image_path)  # this is a PIL image
            x = img_to_array(img)  # this is a Numpy array with shape (3, 150, 150)
            x = x.reshape((1,) + x.shape)  # this is a Numpy array with shape (1, 3, 150, 150)
            i = 0
            for batch in datagen.flow(x, batch_size=1,
                                      save_to_dir=subject_dir_path, save_prefix=image_name, save_format='jpg'):
                i += 1
                if i > 10:
                    break  # otherwise the generator would loop indefinitely


def read_img():
    dirs = os.listdir(path)
    imgs = []
    labels = []
    for dir_name in dirs:
        lable = int(dir_name)
        subject_dir_path = path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            #            image = cv2.imread(image_path)
            image, landmarks2 = read_im_and_landmarks(image_path)
            if image is None:
                continue
            if len(landmarks2) == 1:
                continue
            print(image_path)
            M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                           landmarks2[ALIGN_POINTS])
            image = warp_im(image, M, im1.shape)
            # img = cv2.imread(path)
            #            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            image = cv2.resize(image, (96, 96))
            h, w, c = image.shape
            #            img_col = img_gray.reshape(h * w)
            imgs.append(image)
            labels.append(lable)
    return np.asarray(imgs, np.float32), np.asarray(labels, np.int32)


def getImageKeras():
    #generate()
    data, label = read_img()
    if os.path.isfile("app/data_keras.npy"):
        os.remove("app/data_keras.npy")
    if os.path.isfile("app/label_keras.npy"):
        os.remove("app/label_keras.npy")
    np.save("app/data_keras.npy", data)
    np.save("app/label_keras.npy", label)
