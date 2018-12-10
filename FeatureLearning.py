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

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=10, suppress=True, threshold=np.inf, linewidth=100)

def getImageFeature():
    imageData = glob.glob("img/**/*.jpg")
    for fileName in imageData:
        print(fileName)
    return np.zeros([50, 120], dtype=float)

def getTrainingdata():
    trainingFiles = glob.glob("TrainingData/*.csv")
    print(trainingFiles)

    trainingLabel = np.empty(0, dtype=int)
    trainingFeature = np.empty((0, 120), dtype=float)

    #画像特徴量の取得
    imageFeatues = getImageFeature()
    
    #それぞれの教師データを取得
    for fileName in trainingFiles:
        matrix = np.loadtxt(fileName, delimiter=",")

        #特徴量の追加
        trainingFeature = np.append(trainingFeature, imageFeatues, axis=0)

        #ラベルの追加
        for imageIdx in range(50):
            selected = matrix[imageIdx, :]
            trainingLabel = np.append(trainingLabel, selected, axis=0)
    
    return trainingFeature, trainingLabel

feature, label = getTrainingdata()

sys.exit()

#------------------------------------------------------------------------


#------------------------------------------------------------------------

