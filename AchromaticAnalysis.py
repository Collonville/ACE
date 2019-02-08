import copy
import glob
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

import cv2

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=10, suppress=True, threshold=np.inf, linewidth=100)

#理想白色の制御値
Wtarget = np.array([0.9075, 0.6791, 0.4823])

def getImageRGBFromPath(filePath):
    inputImg = cv2.imread(filePath, cv2.IMREAD_COLOR)

    #正規化と整形
    inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
    rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))

    return rgb, inputImg.shape[0], inputImg.shape[1]

imgInputPath = "outimg/ACE2/ACEMethod1/BlendImage/"
imgOutputPath = "outimg/ACE2/ACEMethod1/FinalSignalImage/"
imgFileName = sys.argv[1] 

#ブレンドした画像の読み込み
RGB, imgH, imgW = getImageRGBFromPath(imgInputPath + imgFileName + ".jpg")
inputRGB        = np.copy(RGB)
maskRGB         = np.copy(RGB)
analyzedRGB     = np.copy(RGB)

HSV = colour.RGB_to_HSV(RGB)

#低彩度値の画素を抽出
achromaticWhitePixelBool = np.where((HSV[:, 2] >= 0.80) & (HSV[:, 1] <= 0.45))
achromaticBlackPixelBool = np.where(HSV[:, 2] <= 0.2)

#対象領域の着色
maskRGB[achromaticWhitePixelBool] = np.array([1, 1, 0])
maskRGB[achromaticBlackPixelBool] = np.array([1, 0, 0])

#無彩色変換
analyzedRGB[achromaticWhitePixelBool] = HSV[achromaticWhitePixelBool, 2].reshape((-1, 1)) * np.transpose(Wtarget)
analyzedRGB[achromaticBlackPixelBool] = HSV[achromaticBlackPixelBool, 2].reshape((-1, 1)) * np.transpose(Wtarget)

#-----------------------------------------------
inputRGB    = inputRGB.reshape((imgH, imgW, 3))
maskRGB     = maskRGB.reshape((imgH, imgW, 3))
analyzedRGB = analyzedRGB.reshape((imgH, imgW, 3))

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.imshow(inputRGB)
ax1.set_title("Input image")

ax2 = fig.add_subplot(132)
ax2.imshow(maskRGB)
ax2.set_title("Achromatic pixel(Red:blackish, Yellow:whitish)")

ax3 = fig.add_subplot(133)
ax3.imshow(analyzedRGB)
ax3.set_title("Achromatic analysised image")

im = Image.fromarray(np.uint8(analyzedRGB * 255))
im.save(imgOutputPath + imgFileName + "_Final.jpg", quality=100)

plt.show()
