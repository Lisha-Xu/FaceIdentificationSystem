import pickle
import cv2
import os
import numpy as np
from app.dlib_test import read_im_and_landmarks,transformation_from_points, warp_im
from keras.models import load_model
import tensorflow as tf
from sklearn.preprocessing import LabelBinarizer
from app.GetImage import LBP

SCALE_FACTOR = 1
FEATHER_AMOUNT = 11
PREDICTOR_PATH = "F:/Identification_System/shape_predictor_68_face_landmarks.dat"
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

with open('app/svm.pickle', 'rb') as fr:
    clf = pickle.load(fr)

with open('app/pca.pickle', 'rb') as fr:
    pca = pickle.load(fr)


im1, landmarks1 = read_im_and_landmarks("app/static/uploads/1.bmp")
def judge(test_path):
    test2 = []
    test, landmarks2 = read_im_and_landmarks(test_path)
    M = transformation_from_points(landmarks1[ALIGN_POINTS],
                                   landmarks2[ALIGN_POINTS])
    test = warp_im(test, M, im1.shape)
    if not os.path.isfile("app/face.h5"):
        face = cv2.cvtColor(test, cv2.COLOR_BGR2GRAY)

        #face,rect = detect_face(face)
        face = cv2.resize(face,(64,64))
        face = face.reshape(64*64)
        print(face)
        face = LBP(face)
        face = face/255
        test2.append(face)
        xx= pca.transform(test2)
        x = clf.predict(xx)
    else:
        G=tf.Graph()
        label = np.load("app/label_keras.npy")
        with G.as_default():
            sess = tf.Session(graph=G)
            with sess.as_default():
                face = cv2.resize(test, (96, 96))
                face = face/255
                test2.append(face)
                data =np.asarray(test2, np.float32)
                model = load_model("app/face.h5")
                x = model.predict(data)
                lb = LabelBinarizer()
                lb.fit(label)
                x = lb.inverse_transform(x)
    x = str(x)
    print(x)
    return x+'.jpg'

#t=judge("F:/Identification_System/app/static/uploads/check/75.jpg")