import copy
import itertools
import math
import sys

import colour
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numba
import numpy as np
import win_unicode_console
from colour.models import *
from colour.plotting import *
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageColor
from scipy import linalg, signal
from scipy.fftpack import *
from scipy.optimize import *
from skimage import filters
from sklearn.metrics import mean_squared_error
from sympy import *
from sympy.matrices import *
from scipy.stats import entropy
import cv2
from scipy.stats import kurtosis, skew
import glob
import codecs
import os

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

np.save("AllFeatures", feature)
np.save("Labels", label)

print(feature.shape)
print(label.shape)

#------------------------------------------------------------------------


#------------------------------------------------------------------------

