
import glob
import itertools
import math
import sys

import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import win_unicode_console
from colour.models import *
from colour.plotting import *
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageColor

import ImageFeature

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=10, suppress=True, threshold=np.inf, linewidth=100)

#------------------------------------------------------------------------
fileName = "water-lily-3784022__340"
inputImgPath = "outimg/continuity_hue/All/" + fileName

intercept = np.load("intercept.npy")
coef = np.load("coef.npy") #多重配列になっていたから先頭だけ抽出

imageFeature = ImageFeature.ImageFeature()
propability = np.empty(0)

for it in range(100):
    #Pathに日本語が含まれるとエラー
    inputImg = cv2.imread(inputImgPath + "_" + str(it) + ".jpg", cv2.IMREAD_COLOR)

    #正規化と整形
    inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
    rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))

    feature = imageFeature.getImageFeatureFromRGB(rgb)

    feature[np.isnan(feature)] = 0

    propability = np.r_[propability, 1. / (1. + np.exp(intercept + np.dot(coef, feature)))]

print("Max iter=%d, propability=%f" % (np.argmax(propability), np.max(propability)))
plt.plot(propability)
plt.plot(np.argmax(propability), np.max(propability), marker='o', color='r')
plt.xlabel("Iter")
plt.ylabel("Propability")
plt.grid(True)
plt.show()
