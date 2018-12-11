import glob
import re
import sys

import colour
import cv2
import numpy as np
import win_unicode_console
from colour.models import *
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import entropy, kurtosis, skew
from skimage import data, filters
from skimage.measure import *
from sklearn.metrics import mean_squared_error

win_unicode_console.enable()

#表示のフォーマットを定義
np.seterr(all='warn', over='raise')
np.set_printoptions(precision=8, suppress=True, threshold=np.inf, linewidth=150)

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
    sat1 = colour.RGB_to_HSL(rgb1)[:, 1]
    sat2 = colour.RGB_to_HSL(rgb2)[:, 1]

    return mean_squared_error(sat1, sat2)

def getImageMeasure(pixels1, pixels2):
    NRMSE = compare_nrmse(pixels1, pixels2)
    PSNR = compare_psnr(pixels1, pixels2)
    SSIM = compare_ssim(pixels1, pixels2, multichannel=True)
    
    return [NRMSE, PSNR, SSIM]

def getBrightnessMeasure(rgb):
    Y = colour.RGB_to_YCbCr(rgb)[:, 0]
    lum = colour.RGB_to_HSL(rgb)[:, 2]

    return np.r_[np.mean(Y), np.var(Y), np.min(Y), np.max(Y), np.mean(lum), np.var(lum), np.min(lum), np.max(lum)]

def getContasrtMeasure(rgb):
    lum = colour.RGB_to_HSL(rgb)[:, 2]

    #正規化
    lum = (lum - np.min(lum)) / (np.max(lum) - np.min(lum))

    return np.var(lum)

def getSaturationMeasure(rgb):
    sat = colour.RGB_to_HSL(rgb)[:, 1]

    return np.r_[np.mean(sat), np.var(sat), np.min(sat), np.max(sat)]

#Measuring colourfulness in natural images [D.hasler, S.Susstrunk]
def getColourFulness(rgb):
    rg = rgb[:, 0] - rgb[:, 1]
    yb = 0.5 * (rgb[:, 0] + rgb[:, 1]) - rgb[:, 2]

    mean_rgyb = np.sqrt(np.mean(rg)**2 + np.mean(yb)**2)
    std_rgyb = np.sqrt(np.std(rg)**2 + np.std(yb)**2)

    return std_rgyb + 0.3 * mean_rgyb

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

#A new universal colour image fidelity metric
#x:Original Image, y:Processed Image, 1 channel
def FidelityMetric(x, y):
    xMean = np.mean(x)
    yMean = np.mean(y)

    #不偏分散
    xVar = np.var(x, ddof=1)
    yVar = np.var(y, ddof=1)

    xyCov = np.cov([x, y], ddof=1, rowvar=False)[0, 0]

    Q1 = xyCov / (np.sqrt(xVar) * np.sqrt(yVar))
    Q2 = (2 * xMean * yMean) / (xMean**2 + yMean**2)
    Q3 = (2 * np.sqrt(xVar) * np.sqrt(yVar)) / (xVar + yVar)

    return (Q1 * Q2 * Q3)

#https://www.pyimagesearch.com/2015/03/23/sliding-windows-for-object-detection-with-python-and-opencv/
def SlidingWindow(imgX, imgY, stepSize, windowSize):
    for y in range(0, inputImg.shape[0], stepSize):
        for x in range(0, inputImg.shape[1], stepSize):
            yield imgX[y: y + windowSize, x: x + windowSize, :], imgY[y: y + windowSize, x: x + windowSize, :]

def ColorFidelityMetric(rgbX, rgbY):
    M = 0
    Q = 0

    #W * H * 3に変形
    rgbX_ = np.reshape(rgbX, (inputImg.shape[0], inputImg.shape[1], 3))
    rgbY_ = np.reshape(rgbY, (inputImg.shape[0], inputImg.shape[1], 3))

    #RGB-->Lab
    XYZ_X = colour.sRGB_to_XYZ(rgbX_)
    XYZ_Y = colour.sRGB_to_XYZ(rgbY_)
    Lab_X = colour.XYZ_to_Lab(XYZ_X)
    Lab_Y = colour.XYZ_to_Lab(XYZ_Y)

    #ウィンドウサイズごとのFidelityMetricを計算
    for (cropX, cropY) in SlidingWindow(Lab_X, Lab_Y, 1, 8):
        # if the window does not meet our desired window size, ignore it
        if cropX.shape[0] != 8 or cropX.shape[1] != 8:
            continue 

        #オリジナル画像のクリップ
        cropL_X = cropX[:, :, 0].flatten()
        cropa_X = cropX[:, :, 1].flatten()
        cropb_X = cropX[:, :, 2].flatten()

        #加工画像のクリップ
        cropL_Y = cropY[:, :, 0].flatten()
        cropa_Y = cropY[:, :, 1].flatten()
        cropb_Y = cropY[:, :, 2].flatten()

        Ql = FidelityMetric(cropL_X, cropL_Y)
        Qa = FidelityMetric(cropa_X, cropa_Y)
        Qb = FidelityMetric(cropb_X, cropb_Y)

        Q += np.sqrt(Ql**2 + Qa**2 + Qb**2)

        M += 1

    return Q / M

numbers = re.compile(r'(\d+)')
def numericalSort(value):
    parts = numbers.split(value)
    parts[1::2] = map(int, parts[1::2])
    return parts

#------------------------------------------------------------------------
#globだけではファイルの順列は保証されないためkey=numericalSortを用いる
imagesPath = sorted(glob.glob('outimg/continuity_hue/All/*.jpg'), key=numericalSort)

featuresWithPath={}

for fileName in imagesPath:
    inputImg = cv2.imread(fileName, cv2.IMREAD_COLOR)

    #正規化と整形
    inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
    rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))

    moment, cov = getColorMoment(rgb)
    aveGrads = getAveGrad(rgb)
    brightness = getBrightnessMeasure(rgb)
    contrast = getContasrtMeasure(rgb)
    SatMeasures = getSaturationMeasure(rgb)
    colorfulness = getColourFulness(rgb)
    naturalness = getNaturalness(rgb)
    #naturalnessの1301-1309でNanが発生
    allFeatures = np.r_[moment, cov.flatten(), aveGrads, brightness, contrast, SatMeasures, colorfulness, naturalness]

    #辞書型でファイル名と特徴量を紐づけ
    featuresWithPath[fileName] = allFeatures

#numpyの保存形式で保存
#https://stackoverflow.com/questions/40219946/python-save-dictionaries-through-numpy-save
np.save("ImageFeatures", featuresWithPath)

