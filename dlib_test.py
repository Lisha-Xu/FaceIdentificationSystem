# -*- coding: utf-8 -*-
import dlib
import numpy as np
import cv2
#predictor_path = "F:/biyesheji/shape_predictor_5_face_landmarks.dat"
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
ALIGN_POINTS = (LEFT_BROW_POINTS + RIGHT_EYE_POINTS + LEFT_EYE_POINTS +
                RIGHT_BROW_POINTS + NOSE_POINTS + MOUTH_POINTS)

detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(PREDICTOR_PATH)


class TooManyFaces(Exception):
    pass


class NoFaces(Exception):
    pass


def get_landmarks(im):
    rects = detector(im, 1)
    if len(rects) > 1:
       #raise TooManyFaces
        return [0]
    if len(rects) == 0:
       #raise NoFaces
        return [0]
    return np.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])


def transformation_from_points(points1, points2):
    """
    Return an affine transformation [s * R | T] such that:
        sum ||s*R*p1,i + T - p2,i||^2
    is minimized.
    """
    # Solve the procrustes problem by subtracting centroids, scaling by the
    # standard deviation, and then using the SVD to calculate the rotation. See
    # the following for more details:
    #   https://en.wikipedia.org/wiki/Orthogonal_Procrustes_problem

    points1 = points1.astype(np.float64)
    points2 = points2.astype(np.float64)

    c1 = np.mean(points1, axis=0)
    c2 = np.mean(points2, axis=0)
    points1 -= c1
    points2 -= c2

    s1 = np.std(points1)
    s2 = np.std(points2)
    points1 /= s1
    points2 /= s2

    U, S, Vt = np.linalg.svd(points1.T * points2)

    # The R we seek is in fact the transpose of the one given by U * Vt. This
    # is because the above formulation assumes the matrix goes on the right
    # (with row vectors) where as our solution requires the matrix to be on the
    # left (with column vectors).
    R = (U * Vt).T

    return np.vstack([np.hstack(((s2 / s1) * R,
                                       c2.T - (s2 / s1) * R * c1.T)),
                         np.matrix([0., 0., 1.])])


def read_im_and_landmarks(fname):
    im = cv2.imread(fname, cv2.IMREAD_COLOR)
    if im is None:
        return None, 1
    im = cv2.resize(im, (im.shape[1] * SCALE_FACTOR,
                         im.shape[0] * SCALE_FACTOR))
    s = get_landmarks(im)
    return im, s


def warp_im(im, M, dshape):
    output_im = np.zeros(dshape, dtype=im.dtype)
    cv2.warpAffine(im,
                   M[:2],
                   (dshape[1], dshape[0]),
                   dst=output_im,
                   borderMode=cv2.BORDER_TRANSPARENT,
                   flags=cv2.WARP_INVERSE_MAP)
    return output_im

'''
im1, landmarks1 = read_im_and_landmarks("ORL/1/1.bmp")
im2, landmarks2 = read_im_and_landmarks("zhaopian/68/1029.png")

M = transformation_from_points(landmarks1[ALIGN_POINTS],
                               landmarks2[ALIGN_POINTS])

warped_im2 = warp_im(im2, M, im1.shape)
cv2.imshow('img', warped_im2)
cv2.waitKey(0)
'''

'''
# 多张脸使用
def get_landmarks_m(im):
    dets = detector(im, 1)
    # 脸的个数
    print("Number of faces detected: {}".format(len(dets)))
    for i in range(len(dets)):
        facepoint = np.array([[p.x, p.y] for p in predictor(im, dets[i]).parts()])
        for i in range(5):
            im[facepoint[i][1]][facepoint[i][0]] = [232,28,8]
    return im
'''