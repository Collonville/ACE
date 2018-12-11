import codecs
import copy
import glob
import itertools
import os
import sys

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numba
import numpy as np
import win_unicode_console
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg, signal
from scipy.optimize import *
from sklearn.linear_model import LogisticRegression

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=10, suppress=True, threshold=np.inf, linewidth=100)

def exportImageFeature():
    imageFeatures = np.load("ImageFeatures.npy")

    features = []
    
    for fileName in [os.path.basename(r) for r in glob.glob('img/**/*.jpg')]:
        #拡張子とファイル名を分割
        filename = fileName.split('.')

        for index in range(100):
            #インデックス値を挿入
            pathKey = filename[:][0] + "_" + str(index) + "." + filename[:][1]

            #特徴量の取得
            imageFeature = imageFeatures.item().get('outimg/continuity_hue/All\\' + pathKey)
            #imageFeature = np.reshape(imageFeature, (1, 37))
            #print(imageFeature.shape)
            features.append(imageFeature)

    np.save("ImageFeaturesSorted", features)
    
    return np.zeros([50, 120], dtype=float)

def getTrainingdata():
    trainingFiles = glob.glob("TrainingData/*.csv")

    #exportImageFeature()
    #画像特徴量の取得([5000, 特徴量次元数])
    imageFeatues = np.load("ImageFeaturesSorted.npy")

    trainingLabel = np.empty(0, dtype=int)
    trainingFeature = np.empty((0, imageFeatues.shape[1]))
    
    #それぞれの教師データを取得
    for fileName in trainingFiles:
        print(fileName)
        matrix = np.loadtxt(fileName, delimiter=",")

        #特徴量の追加
        trainingFeature = np.r_[trainingFeature, imageFeatues]

        #ラベルの追加
        for imageIdx in range(50):
            selected = matrix[imageIdx, :]
            trainingLabel = np.append(trainingLabel, selected, axis=0)
    
    return trainingFeature, trainingLabel

feature, label = getTrainingdata()

feature[np.isnan(feature)] = 0

np.save("AllFeatures", feature)
np.save("Labels", label)

#------------------------------------------------------------------------
lr = LogisticRegression(tol=1e-5, max_iter=300).fit(feature[0:20000, :], label[0:20000])

print (lr.score(feature, label))
print(lr.intercept_)
print(lr.coef_)
print (lr.score(feature[20000:24000, :], label[20000:24000]))





#------------------------------------------------------------------------