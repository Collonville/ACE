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
from skimage import data
from skimage.measure import *
from sklearn.metrics import mean_squared_error
import pandas as pd
import seaborn as sns

import cv2



def getColorMoment(img, BIN_NUM=50):
    #www.kki.yamanashi.ac.jp/~ohbuchi/courses/2013/sm2013/pdf/sm13_lect01_20131007.pdf

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
    entropy1 = shannon_entropy(pixels[:, 0])
    entropy2 = shannon_entropy(pixels[:, 1])
    entropy3 = shannon_entropy(pixels[:, 2])
    '''
    entropy1 = entropy(np.histogram(pixels[:, 0], bins=BIN_NUM)[0], base=2)
    entropy2 = entropy(np.histogram(pixels[:, 1], bins=BIN_NUM)[0], base=2)
    entropy3 = entropy(np.histogram(pixels[:, 2], bins=BIN_NUM)[0], base=2)
    '''
    return np.r_[entropy1, entropy2, entropy3]

def getDistance(pixels1, pixels2):
    lab1 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(pixels1))
    lab2 = colour.XYZ_to_Lab(colour.sRGB_to_XYZ(pixels2))
    dist = mean_squared_error(lab1, lab2)

    #dist = np.sqrt((pixels1[:, 0] - pixels2[:, 0])**2 + (pixels1[:, 1] - pixels2[:, 1])**2 + (pixels1[:, 2] - pixels2[:, 2])**2)
    #dist = np.sum(dist)

    return dist

def getImageMeasure(pixels1, pixels2):
    NRMSE = compare_nrmse(pixels1, pixels2)
    PSNR = compare_psnr(pixels1, pixels2)
    SSIM = compare_ssim(pixels1, pixels2, multichannel=True)
    
    return [NRMSE, PSNR, SSIM]

#Measuring colourfulness in natural images [D.hasler, S.Susstrunk]
def getColourFulness(rgb):
    rg = rgb[:, 0] - rgb[:, 1]
    yb = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]

    mean_rgyb = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    var_rgyb = np.sqrt(np.var(rg)**2 + np.var(yb)**2)

    return var_rgyb + 0.3 * mean_rgyb

def getNaturalness(rgb):
    XYZ = colour.sRGB_to_XYZ(rgb)
    LUV = colour.XYZ_to_Luv(XYZ)


win_unicode_console.enable()
#------------------------------------------------------------------------

#連続画像から特徴を計算する
BY_CONTINUS_IMAGE = True
COLOR_SPACE = "HSV"

argv = sys.argv

imgName = argv[1]

if BY_CONTINUS_IMAGE:
    MAX_ITER = 100

    moments = np.zeros((MAX_ITER, 12), dtype='float64')
    entropys = np.zeros((MAX_ITER, 3), dtype='float64')
    aveGrads = np.zeros(MAX_ITER, dtype='float64')
    colorDist = np.zeros(MAX_ITER, dtype='float64')
    measures = np.zeros((MAX_ITER, 3), dtype='float64')
    cf = np.zeros(MAX_ITER, dtype='float64')

    for it in range(MAX_ITER):
        path = "outimg/continuity/" + imgName + "_" + str(it) + ".jpg"
        inputImg = cv2.imread(path, cv2.IMREAD_COLOR)

        #正規化と整形
        inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
        rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))
        
        #初期画像の保存
        if(it == 0):
            initialImg = rgb

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
        colorDist[it] = getDistance(initialImg, rgb)
        measures[it] = getImageMeasure(initialImg, rgb)
        cf[it] = getColourFulness(rgb)
else:
    feature = np.load("features/" + imgName + "_Features.npy")
    momentFeature = np.load("features/" + imgName + "_MomentFeatures.npy")

#------------------------------------------------------------------------
#可視化
sns.lineplot(data=pd.DataFrame(cf))
plt.show()

sys.exit()

sns.set(style="darkgrid")
fig, axes = plt.subplots(nrows=2, ncols=4)
ax = axes.ravel()

fig.suptitle('Features of ' + imgName + ".jpg Color Space:" + COLOR_SPACE, fontsize=12, fontweight='bold')

#平均勾配
ax[0].set_ylim(0.0, 1.0)
sns.lineplot(data=pd.DataFrame(aveGrads), ax=ax[0])


#エントロピー
ax[1].set_title(COLOR_SPACE + " Entropy")
df = pd.DataFrame({"Iter":range(MAX_ITER), "Hue": entropys[:, 0], "Saturation":entropys[:, 1], "Luminance": entropys[:, 2]})
sns.lineplot(data=df, x="Iter", y="Hue", label="Hue", ax=ax[1])
sns.lineplot(data=df, x="Iter", y="Saturation", label="Saturation", ax=ax[1])
sns.lineplot(data=df, x="Iter", y="Luminance", label="Luminance", ax=ax[1])


#平均
ax[2].plot(moments[:, 0], label="Red")
ax[2].plot(moments[:, 4], label="Green")
ax[2].plot(moments[:, 8], label="Blue")
ax[2].set_title(COLOR_SPACE + " Mean")
ax[2].set_xlabel("Iter")
ax[2].set_ylim(0.0, 1.0)
ax[2].legend(loc="upper right")

#分散
ax[3].plot(moments[:, 1], label="Red")
ax[3].plot(moments[:, 5], label="Green")
ax[3].plot(moments[:, 9], label="Blue")
ax[3].set_title(COLOR_SPACE + " Variance")
ax[3].set_xlabel("Iter")
ax[3].legend()

#歪度
ax[4].plot(moments[:, 2], label="Red")
ax[4].plot(moments[:, 6], label="Green")
ax[4].plot(moments[:, 10], label="Blue")
ax[4].set_title(COLOR_SPACE + " Skweness")
ax[4].set_xlabel("Iter")
ax[4].legend()

#尖度
ax[5].plot(moments[:, 3], label="Red")
ax[5].plot(moments[:, 7], label="Green")
ax[5].plot(moments[:, 11], label="Blue")
ax[5].set_title(COLOR_SPACE + " kurtosis")
ax[5].set_xlabel("Iter")
ax[5].legend()

ax[6].plot(colorDist)
ax[6].set_title("Euclid Distance from init Img")
ax[6].set_xlabel("Iter")
ax[6].legend()

#PSNRは類似度が全く一緒だとlog0-->無限になるため、正規化する際は1つインデックスをずらず
ax[7].plot(measures[:, 0], label="NRMSE")
ax[7].plot(measures[:, 1] / np.max(measures[1:, 1]), label="PSNR(Normed)")
ax[7].plot(measures[:, 2], label="SSIM")
ax[7].set_title("Measures")
ax[7].set_xlabel("Iter")

ax[7].legend()

#plt.tight_layout()
plt.show()
