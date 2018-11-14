import sys

import colour
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import win_unicode_console
from colour.models import *
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageColor
from scipy.stats import entropy, kurtosis, skew
from skimage import data, filters
from skimage.measure import *
from sklearn.metrics import mean_squared_error

win_unicode_console.enable()

#表示のフォーマットを定義
np.seterr(all='warn', over='raise')
np.set_printoptions(precision=8, suppress=True, threshold=np.inf, linewidth=100)

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
    dist = mean_squared_error(pixels1, pixels2)

    #dist = np.sqrt((pixels1[:, 0] - pixels2[:, 0])**2 + (pixels1[:, 1] - pixels2[:, 1])**2 + (pixels1[:, 2] - pixels2[:, 2])**2)
    #dist = np.sum(dist)

    return dist
def getSatDistance(rgb1, rgb2):
    sat1 = colour.RGB_to_HSL(rgb1)[1]
    sat2 = colour.RGB_to_HSL(rgb2)[1]

    return mean_squared_error(sat1, sat2)

def getImageMeasure(pixels1, pixels2):
    NRMSE = compare_nrmse(pixels1, pixels2)
    PSNR = compare_psnr(pixels1, pixels2)
    SSIM = compare_ssim(pixels1, pixels2, multichannel=True)
    
    return [NRMSE, PSNR, SSIM]

def getBrightnessMeasure(rgb):
    Y = colour.RGB_to_YCbCr(rgb)[0]
    lum = colour.RGB_to_HSL(rgb)[2]

    return np.c_[np.mean(Y), np.var(Y), np.min(Y), np.max(Y), np.mean(lum), np.var(lum), np.min(lum), np.max(lum)]

def getContasrtMeasure(rgb):
    lum = colour.RGB_to_HSL(rgb)[2]

    #正規化
    lum = (lum - np.min(lum)) / (np.max(lum) - np.min(lum))

    return np.var(lum)

def getSaturationMeasure(rgb):
    sat = colour.RGB_to_HSL(rgb)[1]

    return np.c_[np.mean(sat), np.var(sat), np.min(sat), np.max(sat)]

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

    #彩度と色相の計算(色相は0-2pi-->角度(0-360)に変換)
    sat = np.sqrt(LUV[:, 1]**2 + LUV[:, 2]**2) / 100
    hue = np.arctan2(LUV[:, 2], LUV[:, 1])
    hue[hue < 0] += 2 * np.pi

    LHS = np.c_[LUV[:, 0], np.rad2deg(hue), sat]

    #Thresolding L and S components
    LHS = LHS[np.where((LHS[:, 0] >= 20) & (LHS[:, 0] <= 80) & (LHS[:, 2] >= 0.1))]

    #Calcurate average and pixel num of saturation value
    skin  = LHS[np.where((LHS[:, 1] >= 25 ) & (LHS[:, 1] <= 70 )), 2]
    grass = LHS[np.where((LHS[:, 1] >= 95 ) & (LHS[:, 1] <= 135)), 2]
    sky   = LHS[np.where((LHS[:, 1] >= 185) & (LHS[:, 1] <= 260)), 2]

    if(skin.shape[1] == 0):
        n_skin = 0
        S_skin = 0
    else:
        n_skin = skin.shape[1]
        S_skin = np.mean(skin)

    if(grass.shape[1] == 0):
        n_grass = 0
        S_grass = 0
    else:   
        n_grass = grass.shape[1]
        S_grass = np.mean(grass)

    if(sky.shape[1] == 0):
        n_sky = 0
        S_sky = 0
    else:
        n_sky = sky.shape[1]
        S_sky = np.mean(sky)

    #Calcurate local CNI value
    N_skin = np.power(np.exp(-0.5 * ((S_skin - 0.76) / 0.52)**2), 4)
    N_grass = np.exp(-0.5 * ((S_grass - 0.81) / 0.53)**2)
    N_sky = np.exp(-0.5 * ((S_sky - 0.43) / 0.22)**2)

    return (n_skin * N_skin + n_grass * N_grass + n_sky * N_sky) / (n_skin + n_grass + n_sky)



#------------------------------------------------------------------------

#連続画像から特徴を計算する
COLOR_SPACE = "HSV"

argv = sys.argv

imgName = argv[1]

MAX_ITER = 100

moments = np.zeros((MAX_ITER, 12), dtype='float64')
entropys = np.zeros((MAX_ITER, 3), dtype='float64')
aveGrads = np.zeros(MAX_ITER, dtype='float64')
colorDist = np.zeros(MAX_ITER, dtype='float64')
measures = np.zeros((MAX_ITER, 3), dtype='float64')
SatMeasures = np.zeros((MAX_ITER, 4), dtype='float64')
colorfulness = np.zeros(MAX_ITER, dtype='float64')
naturalness = np.zeros(MAX_ITER, dtype='float64')
contrast = np.zeros(MAX_ITER, dtype='float64')
brightness = np.zeros((MAX_ITER, 8), dtype='float64')
satDist = np.zeros(MAX_ITER, dtype='float64')

for it in range(MAX_ITER):
    path = "outimg/continuity/" + imgName + "_" + str(it) + ".jpg"
    inputImg = cv2.imread(path, cv2.IMREAD_COLOR)

    #正規化と整形
    inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
    rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))
    
    #初期画像の保存
    if(it == 0):
        initialRGB = rgb

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
    colorDist[it] = getDistance(initialRGB, rgb)
    measures[it] = getImageMeasure(initialRGB, rgb)

    SatMeasures[it] = getSaturationMeasure(rgb)
    colorfulness[it] = getColourFulness(rgb)
    naturalness[it] = getNaturalness(rgb)
    contrast[it] = getContasrtMeasure(rgb)
    brightness[it] = getBrightnessMeasure(rgb)
    satDist[it] = getSatDistance(rgb, initialRGB)


#------------------------------------------------------------------------
#可視化
sns.set_style("whitegrid", {'grid.linestyle': '--'})
fig, axes = plt.subplots(nrows=2, ncols=4)
ax = axes.ravel()

fig.suptitle('Features of ' + imgName + ".jpg Color Space:" + COLOR_SPACE, fontsize=12, fontweight='bold')

#平均勾配
ax[0].plot(contrast, label="Contasrt var")
ax[0].plot(aveGrads, label="Average Gradient")
ax[0].set_ylim(0.0, 1.1)
ax[0].set_xlabel("Iter")
ax[0].legend()

ax[1].set_title("Colorfulness & Naturalness")
ax[1].plot(colorfulness, label="Colorfulness")
ax[1].plot(naturalness, label="Naturalness")
ax[1].plot(colorfulness + naturalness, label="C + N")
ax[1].set_ylim(0.0, 1.1)
ax[1].set_xlabel("Iter")
ax[1].legend()

#エントロピー
'''
ax[1].set_title(COLOR_SPACE + " Entropy")
df = pd.DataFrame({"Iter":range(MAX_ITER), "Hue": entropys[:, 0], "Saturation":entropys[:, 1], "Luminance": entropys[:, 2]})
sns.lineplot(data=df, x="Iter", y="Hue", label="Hue", ax=ax[1])
sns.lineplot(data=df, x="Iter", y="Saturation", label="Saturation", ax=ax[1])
sns.lineplot(data=df, x="Iter", y="Luminance", label="Luminance", ax=ax[1])
'''
ax[2].set_title("Saturation Measure")
ax[2].plot(SatMeasures[:, 0], label="Mean")
ax[2].plot(SatMeasures[:, 1], label="Var")
ax[2].plot(SatMeasures[:, 2], label="Min")
ax[2].plot(SatMeasures[:, 3], label="Max")
ax[2].legend()
ax[2].set_xlabel("Iter")
ax[2].set_ylim(0.0, 1.1)

ax[3].set_title("Y Measure")
ax[3].plot(brightness[:, 0], label="Measn")
ax[3].plot(brightness[:, 1], label="Var")
ax[3].plot(brightness[:, 2], label="Min")
ax[3].plot(brightness[:, 3], label="Max")
ax[3].set_xlabel("Iter")
ax[3].set_ylim(0.0, 1.1)
ax[3].legend()

ax[4].set_title("L Measure")
ax[4].plot(brightness[:, 4], label="Mean")
ax[4].plot(brightness[:, 5], label="Var")
ax[4].plot(brightness[:, 6], label="Min")
ax[4].plot(brightness[:, 7], label="Max")
ax[4].set_xlabel("Iter")
ax[4].set_ylim(0.0, 1.1)
ax[4].legend()

'''
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
'''
ax[5].plot(aveGrads + naturalness + 0.8*colorfulness - (1-measures[:, 2]))

ax[5].set_title("Energy")
ax[5].set_xlabel("Iter")
ax[5].legend()

ax[6].plot(colorDist, label="RGB")
ax[6].plot(satDist, label="Satuation")
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
