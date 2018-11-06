import sys

import colour
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import win_unicode_console
from colour.models import *
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageColor
from scipy.stats import entropy, kurtosis, skew
from skimage import filters

import cv2


def getColorMoment(img):
    #www.kki.yamanashi.ac.jp/~ohbuchi/courses/2013/sm2013/pdf/sm13_lect01_20131007.pdf
    img = np.clip(img, 0, 1)

    rChannel = img[:, 0]
    gChannel = img[:, 1]
    bChannel = img[:, 2]

    #平均
    rMean = np.mean(rChannel)
    gMean = np.mean(gChannel)
    bMean = np.mean(bChannel)

    #分散
    rVar = np.var(rChannel)
    gVar = np.var(gChannel)
    bVar = np.var(bChannel)

    #共分散
    #bias=True:画素数で割る、rowvar=False:各チャンネルが列に並んでいるため
    rgbCov = np.cov(img, bias=True, rowvar=False)

    #歪度
    rSkew = skew(rChannel)
    gSkew = skew(gChannel)
    bSkew = skew(bChannel)

    #尖度
    rKurt = kurtosis(rChannel)
    gKurt = kurtosis(gChannel)
    bKurt = kurtosis(bChannel)

    rMoment = [rMean, rVar, rSkew, rKurt]
    gMoment = [gMean, gVar, gSkew, gKurt]
    bMoment = [bMean, bVar, bSkew, bKurt]

    return np.r_[rMoment, gMoment, bMoment], rgbCov

def getAveGrad(pixels):
    #勾配値情報
    pixels_ = np.reshape(pixels, (inputImg.shape[0], inputImg.shape[1], 3))
    redGrad = filters.sobel(pixels_[:, :, 0])
    greGrad = filters.sobel(pixels_[:, :, 1])
    bluGrad = filters.sobel(pixels_[:, :, 2])

    imgGrad = np.sqrt(redGrad**2 + greGrad**2 + bluGrad**2)

    return np.sum(imgGrad) / (inputImg.shape[0] * inputImg.shape[1])

def getEntropy(pixels, BIN_NUM=50):
    entropy1 = entropy(np.histogram(pixels[:, 0], bins=BIN_NUM)[0], base=2)
    entropy2 = entropy(np.histogram(pixels[:, 1], bins=BIN_NUM)[0], base=2)
    entropy3 = entropy(np.histogram(pixels[:, 2], bins=BIN_NUM)[0], base=2)

    return np.r_[entropy1, entropy2, entropy3]

win_unicode_console.enable()
#------------------------------------------------------------------------

#連続画像から特徴を計算する
BY_CONTINUS_IMAGE = True
COLOR_SPACE = "HSL"

argv = sys.argv

imgName = argv[1]

if BY_CONTINUS_IMAGE:
    MAX_ITER = 100

    moments = np.zeros((MAX_ITER, 12), dtype='float64')
    entropys = np.zeros((MAX_ITER, 3), dtype='float64')
    aveGrads = np.zeros(MAX_ITER, dtype='float64')

    for it in range(MAX_ITER):
        path = "outimg/continuity/" + imgName + "_" + str(it) + ".jpg"
        inputImg = cv2.imread(path, cv2.IMREAD_COLOR)

        #正規化と整形
        inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
        rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))

        #色空間の変換
        if COLOR_SPACE is "RGB":
            pixel = rgb
        elif COLOR_SPACE is "HSV":
            pixel = colour.RGB_to_HSV(rgb)
        elif COLOR_SPACE is "HSL":
            pixel = colour.RGB_to_HSL(rgb)

        aveGrads[it] = getAveGrad(pixel)
        entropys[it] = getEntropy(pixel)
        moment, cov = getColorMoment(pixel)
        moments[it] = moment
else:
    feature = np.load("features/" + imgName + "_Features.npy")
    momentFeature = np.load("features/" + imgName + "_MomentFeatures.npy")

#------------------------------------------------------------------------
#可視化
fig, axes = plt.subplots(nrows=2, ncols=3)
ax = axes.ravel()

fig.suptitle('Features of ' + imgName + ".jpg Color Space:" + COLOR_SPACE, fontsize=12, fontweight='bold')

#平均勾配
ax[0].plot(aveGrads)
ax[0].set_title("Average Gradient")
ax[0].grid(True)
ax[0].legend()
ax[0].set_xlabel("Iter")
ax[0].set_ylim(0.0, 1.0)

#エントロピー
ax[1].plot(entropys[:, 0], label="Hue")
ax[1].plot(entropys[:, 1], label="Saturation")
ax[1].plot(entropys[:, 2], label="Luminance")
ax[1].set_title(COLOR_SPACE + " Entropy")
ax[1].set_xlabel("Iter")
ax[1].grid(True)
ax[1].legend(loc="upper right")

#平均
ax[2].plot(moments[:, 0], label="Red")
ax[2].plot(moments[:, 4], label="Green")
ax[2].plot(moments[:, 8], label="Blue")
ax[2].set_title(COLOR_SPACE + " Mean")
ax[2].set_xlabel("Iter")
ax[2].set_ylim(0.0, 1.0)
ax[2].grid(True)
ax[2].legend(loc="upper right")

#分散
ax[3].plot(moments[:, 1], label="Red")
ax[3].plot(moments[:, 5], label="Green")
ax[3].plot(moments[:, 9], label="Blue")
ax[3].set_title(COLOR_SPACE + " Variance")
ax[3].set_xlabel("Iter")
ax[3].grid(True)
ax[3].legend()

#歪度
ax[4].plot(moments[:, 2], label="Red")
ax[4].plot(moments[:, 6], label="Green")
ax[4].plot(moments[:, 10], label="Blue")
ax[4].set_title(COLOR_SPACE + " Skweness")
ax[4].set_xlabel("Iter")
ax[4].grid(True)
ax[4].legend()

#尖度
ax[5].plot(moments[:, 3], label="Red")
ax[5].plot(moments[:, 7], label="Green")
ax[5].plot(moments[:, 11], label="Blue")
ax[5].set_title(COLOR_SPACE + " kurtosis")
ax[5].set_xlabel("Iter")
ax[5].grid(True)
ax[5].legend()

plt.tight_layout()
plt.show()
