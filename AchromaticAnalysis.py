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

def getImageRGBFromPath(filePath):
    inputImg = cv2.imread(filePath, cv2.IMREAD_COLOR)

    #正規化と整形
    inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
    rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))

    return rgb, inputImg.shape[0], inputImg.shape[1]

imgPath= sys.argv[1]

#ブレンドした画像の読み込み
RGB, imgH, imgW = getImageRGBFromPath(imgPath + ".jpg")
inputRGB = np.copy(RGB)

HSV = colour.RGB_to_HSV(RGB)

#低彩度値の画素を抽出
achromaticWhitePixelBool = np.where(HSV[:, 2] > 0.8)
achromaticBlackPixelBool = np.where(HSV[:, 2] < 0.25)

RGB[achromaticWhitePixelBool] = np.array([1, 1, 0])
RGB[achromaticBlackPixelBool] = np.array([1, 0, 0])

#-----------------------------------------------
inputRGB = inputRGB.reshape((imgH, imgW, 3))
RGB = RGB.reshape((imgH, imgW, 3))

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.imshow(inputRGB)
ax1.set_title("Input blend image")

ax2 = fig.add_subplot(132)
ax2.imshow(RGB)
ax2.set_title("Achromatic Pixel")

'''
im = Image.fromarray(np.uint8(blendRGB * 255))
im.save(blendImgOutputPath + imgFilename + "_" + str(iter) + "_k" + str(k).replace('.', '') + "_Blend.jpg", quality=100)
'''
plt.show()



