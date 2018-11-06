import sys

import colour
import cv2
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

def colorMoment(img):
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

def getFeature(img):
    img = np.clip(img, 0, 1)
    img_reshaped = img.reshape((inputImg.shape[0], inputImg.shape[1], 3))

    #勾配値情報
    redGrad = filters.sobel(img_reshaped[:, :, 0])
    greGrad = filters.sobel(img_reshaped[:, :, 1])
    bluGrad = filters.sobel(img_reshaped[:, :, 2])
    imgGrad = np.sqrt(redGrad**2 + greGrad**2 + bluGrad**2)
    aveGrad = np.sum(imgGrad) / (img_reshaped.shape[0] * img_reshaped.shape[1])

    #エントロピー情報
    BIN_NUM = 50
    hsl = colour.RGB_to_HSL(img)

    hueEntropy = entropy(np.histogram(hsl[:, 0].flatten(), bins=BIN_NUM)[0], base=2)
    satEntropy = entropy(np.histogram(hsl[:, 1].flatten(), bins=BIN_NUM)[0], base=2)
    lumEntropy = entropy(np.histogram(hsl[:, 2].flatten(), bins=BIN_NUM)[0], base=2)

    return np.r_[aveGrad, hueEntropy, satEntropy, lumEntropy]

def getEntropy(pixels, BIN_NUM=50):
    entropy1 = entropy(np.histogram(pixels[:, 0].flatten(), bins=BIN_NUM)[0], base=2)
    entropy2 = entropy(np.histogram(pixels[:, 1].flatten(), bins=BIN_NUM)[0], base=2)
    entropy3 = entropy(np.histogram(pixels[:, 2].flatten(), bins=BIN_NUM)[0], base=2)

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
    feature = np.zeros((MAX_ITER, 4), dtype='float64')
    momentFeature = np.zeros((MAX_ITER, 12), dtype='float64')

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

        feature[it] = getFeature(pixel)
        moment, cov = colorMoment(pixel)
        momentFeature[it] = moment
else:
    feature = np.load("features/" + imgName + "_Features.npy")
    momentFeature = np.load("features/" + imgName + "_MomentFeatures.npy")


#------------------------------------------------------------------------
#可視化
fig, axes = plt.subplots(nrows=2, ncols=3)
ax = axes.ravel()

ax[0].plot(feature[:, 0])
ax[0].set_title("Average Gradient")
ax[0].grid(True)
ax[0].legend()
ax[0].set_xlabel("Iter")
ax[0].set_ylim(0.0, 1.0)

#エントロピー
ax[1].plot(feature[:, 1], label="Hue")
ax[1].plot(feature[:, 2], label="Saturation")
ax[1].plot(feature[:, 3], label="Luminance")
ax[1].set_title("HSL Entropy")
ax[1].set_xlabel("Iter")
ax[1].grid(True)
ax[1].legend(loc="upper right")

#平均
ax[2].plot(momentFeature[:, 0], label="Red")
ax[2].plot(momentFeature[:, 4], label="Green")
ax[2].plot(momentFeature[:, 8], label="Blue")
ax[2].set_title(u"RGB Mean")
ax[2].set_xlabel("Iter")
ax[2].set_ylim(0.0, 1.0)
ax[2].grid(True)
ax[2].legend(loc="upper right")

#分散
ax[3].plot(momentFeature[:, 1], label="Red")
ax[3].plot(momentFeature[:, 5], label="Green")
ax[3].plot(momentFeature[:, 9], label="Blue")
ax[3].set_title(u"RGB Variance")
ax[3].set_xlabel("Iter")
ax[3].grid(True)
ax[3].legend()

#歪度
ax[4].plot(momentFeature[:, 2], label="Red")
ax[4].plot(momentFeature[:, 6], label="Green")
ax[4].plot(momentFeature[:, 10], label="Blue")
ax[4].set_title(u"RGB Skweness")
ax[4].set_xlabel("Iter")
ax[4].grid(True)
ax[4].legend()

#尖度
ax[5].plot(momentFeature[:, 3], label="Red")
ax[5].plot(momentFeature[:, 7], label="Green")
ax[5].plot(momentFeature[:, 11], label="Blue")
ax[5].set_title("RGB kurtosis")
ax[5].set_xlabel("Iter")
ax[5].grid(True)
ax[5].legend()

plt.tight_layout()
plt.show()
