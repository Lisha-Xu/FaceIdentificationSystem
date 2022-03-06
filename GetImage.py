from __future__ import print_function
import cv2
import numpy as np
from numpy import *
import os

data_folder_path = "F:/biyesheji/PythonStudyCode/app/static/uploads/users/"
from app.dlib_test import read_im_and_landmarks,transformation_from_points, warp_im

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

from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
datagen = ImageDataGenerator(
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='constant',
    cval = 0)


def generate():
    dirs = os.listdir(data_folder_path)
    for dir_name in dirs:
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        if len(subject_images_names)<10:
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
                    if i > 3:
                        break  # otherwise the generator would loop indefinitely

def minBinary(pixel):
    length = len(pixel)
    zero = ''
    for i in range(length)[::-1]:
        if pixel[i] =='0':
            pixel = pixel[:i]
            zero += '0'
        else:
            return zero+pixel
    if len(pixel) == 0:
        return '0'

def LBP(FaceMat, R=2, P=8):
    Region8_x = [-1,0,1,1,1,0,-1,-1]
    Region8_y = [-1,-1,-1,0,1,1,1,0]
    pi = math.pi
    LBPoperator = zeros(shape(FaceMat))
    #for i in range(shape(FaceMat)[1]):
    face = FaceMat.reshape(64,64)
    W,H = shape(face)
    tempface = zeros([W,H])
    for x in range(R,W-R):
        for y in range(R,H-R):
            repixel = ''
            pixel = int(face[x,y])
            for p in [2,1,0,7,6,5,4,3]:
                p = float(p)
                xp = x + R*cos(2*pi*(p/P))
                yp = y- R*sin(2*pi*(p/P))
                xp = int(xp)
                yp = int(yp)
                if face[xp,yp]>pixel:
                    repixel += '1'
                else:
                    repixel += '0'
            tempface[x,y] = int(minBinary(repixel),base=2)
    LBPoperator = tempface.flatten()
    return LBPoperator

def get_Image():
    dirs = os.listdir(data_folder_path)
    for dir_name in dirs:
        lable = int(dir_name)
        subject_dir_path = data_folder_path + "/" + dir_name
        subject_images_names = os.listdir(subject_dir_path)
        for image_name in subject_images_names:
            if image_name.startswith("."):
                continue
            image_path = subject_dir_path + "/" + image_name
            image, landmarks2 = read_im_and_landmarks(image_path)
            if image is None:
                continue
            if len(landmarks2) == 1:
                continue
            print(image_path)
            M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                           landmarks2[ALIGN_POINTS])
            image = warp_im(image, M, im1.shape)
            #img = cv2.imread(path)
            img_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            img_gray = cv2.resize(img_gray,(64,64))
            h, w = img_gray.shape
            img_col = img_gray.reshape(h * w)
            FaceMat = mat(zeros((h * w)))
            FaceMat = img_col
            LBPoperator = LBP(FaceMat)
            all_data_set.append(LBPoperator)
            #all_data_set.append(FaceMat.flatten())
            all_data_label.append(lable)
    return h,w


all_data_set = []
all_data_label = []

def GetImage():
    generate()
    h,w = get_Image()
    if os.path.isfile("app/data.npy"):
        os.remove("app/data.npy")
    if os.path.isfile("app/label.npy"):
        os.remove("app/label.npy")
    np.save("app/data.npy",all_data_set)
    np.save("app/label.npy",all_data_label)
