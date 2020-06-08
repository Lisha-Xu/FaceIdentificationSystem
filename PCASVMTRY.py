from __future__ import print_function
import os
from time import time
import logging
import matplotlib.pyplot as plt
import cv2
from numpy import *
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.decomposition import PCA
from sklearn.svm import SVC
import pickle
h = 64
w=64

def trainPCA():
    all_data_set = np.load("app/data.npy")
    all_data_label = np.load("app/label.npy")
    X = array(all_data_set)
    X = X/255
    y = array(all_data_label)
    n_samples,n_features = X.shape
    n_classes = len(unique(y))
    target_names = []


    for i in range(1,n_classes+1):
        names = "person" + str(i)
        target_names.append(names)
    '''
    print("Total dataset size:")

    print("n_samples: %d" % n_samples)

    print("n_features: %d" % n_features)
    print("n_classes: %d" % n_classes)
    # split into a training and testing set
    '''
    for i in range(1,30000):
        #print(i)
        X_train, X_test, y_train, y_test = train_test_split(
    
            X, y, test_size=0.25, random_state=i)
        if len(unique(y_train))==len(unique(y_test)):
            break
    '''
    X_train, X_test, y_train, y_test = train_test_split(
       X, y, test_size=0.25, random_state=579)
    '''
    n_components = 55
    '''
    print("Extracting the top %d eigenfaces from %d faces"

          % (n_components, X_train.shape[0]))
    '''
    #t0 = time()
    pca = PCA(n_components=n_components, svd_solver='randomized',
              whiten=True).fit(X_train)
    with open('app/pca.pickle', 'wb') as fw:
        pickle.dump(pca, fw)
    eigenfaces = pca.components_.reshape((n_components, h, w))
    t0 = time()
    X_train_pca = pca.transform(X_train)
    X_test_pca = pca.transform(X_test)
    #print("Fitting the classifier to the training set")
    param_grid = {'C': [1e3, 5e3, 1e4, 5e4, 1e5],
                  'gamma': [0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], }
    clf = GridSearchCV(SVC(kernel='rbf', class_weight='balanced'), param_grid)
    clf = clf.fit(X_train_pca, y_train)
    x = time()-t0
    #print(clf.best_estimator_)
    #print("Predicting people's names on the test set")

    t0 = time()
    y_pred = clf.predict(X_test_pca)
    #print(len(unique(y_pred)))
    #print("done in %0.3fs" % (time() - t0))
    report=(classification_report(y_test, y_pred, digits=4,target_names=target_names))
    report = report.replace('\n','\r\n')
    #print(report)
    #print(n_components)
    #print(confusion_matrix(y_test, y_pred, labels=range(n_classes)))
    with open('app/svm.pickle', 'wb') as fw:
        pickle.dump(clf, fw)
    #if os.path.isfile("app/face.h5"):
    #    os.remove("app/face.h5")
    #print("finish")
    return report,x


#trainPCA()
