import copy
import itertools
import math
import sys

import numpy as np
from scipy import linalg, signal
from scipy.fftpack import *

import colour
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numba
import win_unicode_console
from colour.models import *
from colour.plotting import *
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageColor
from sklearn.metrics import mean_squared_error
from sympy import *
from sympy.matrices import *

win_unicode_console.enable()

np.seterr(all='warn', over='raise')

#表示のフォーマットを定義
np.set_printoptions(precision=8, suppress=True, threshold=np.inf, linewidth=100)

@numba.jit('f8[:](f8[:], f8[:], i8, i8, f4)')
def RIslow(omega, I, width, height, alpha):
    valsum = np.zeros(width * height, dtype='float64')
    for y in range(width * height):
        sums = 0
        for x in range(width * height):
            sums += omega[x] * np.minimum(np.maximum(alpha * (I[x] - I[y]), -1), 1)

        valsum[y] = sums 
    return valsum

def RI(omegaFFT, img):
    valsum = np.zeros(img.size, dtype='float64')
    
    for n in range(degree + 1):
        a = np.zeros(img.size, dtype='float64')
        for m in range(n, degree + 1):
            #print("%d, %d, %d" % (n, m, (m - n + 1)))
            first = 1 if (m - n + 1) % 2 else -1
            a += first * poly[n] * binom_coeff(m, n) * np.power(img, m - n, dtype='float64')
        
        valsum += a * irfft(omegaFFT * rfft(np.power(img, n))).real

    #stretch to [0,1]
    valsum = (valsum - np.min(valsum)) / (np.max(valsum) - np.min(valsum))
    return valsum

def computeOmegaTrans(width, height):
    omegaDistanceFunc = "Euclidean"

    omega = np.zeros(width * height, dtype='float64')

    print("%d, %d" % (height, width))
    if omegaDistanceFunc is "Gaussian":
        sigma = 10
        for y in range(height):
            for x in range(width):
                omega[y * width + x] += np.exp(-(x**2 + y**2) / (2 * sigma**2))
    elif omegaDistanceFunc is "Euclidean":
        for y in range(height):
            for x in range(width):
                omega[y * width + x] = 0 if (x == 0 & y == 0) else 1. / np.sqrt(x**2 + y**2)

    #omegaの合計値が1になるように正規化
    omega = omega / np.sum(omega)

    return rfft(omega), omega

def energy(I, I0, alpha, beta, gamma, width, height):
    first = (alpha * 0.5) * np.sum((I - 0.5) ** 2)
    second = 0
    for y in range(height):
        for x in range(width):
            second += omega[y * width + x] * np.abs(I[x] - I0[y])
    third = (beta * 0.5) * np.sum((I - I0) ** 2)

    return first - second + third

def s(r):
    return np.minimum(np.maximum(alpha * r, -1), 1)

def ITP2ICH(ITP):
    C = np.sqrt(ITP[:, 1]**2 + ITP[:, 2]**2)
    H = np.arctan2(ITP[:, 1], ITP[:, 2])
    
    return np.c_[ITP[:, 0], C, H]

def ICH2ITP(ICH):
    P = ICH[:, 1] * np.cos(ICH[:, 2])
    T = ICH[:, 1] * np.sin(ICH[:, 2])

    return np.c_[ICH[:, 0], T, P]

#Sympyのシンボル定義
r, g, b = symbols("r g b")
r_, g_, b_ = symbols("r_ g_ b_")
L, M, S = symbols("L M S")
I, T, P = symbols("I T P")
I_, T_, P_ = symbols("I_ T_ P_")

#ITP<-->RGB変換への定義
RGB2LMS_Mat = Matrix([
    [1688 / 4096, 2146 / 4096, 262 / 4096],
    [683 / 4096 , 2951 / 4096, 462 / 4096],
    [99 / 4096  , 309 / 4096 , 3688 / 4096]
])
LMS2ITP_Mat = Matrix([
    [0.5         ,  0.5         ,  0          ],
    [6610 / 4096 , -13613 / 4096,  7003 / 4096],
    [17933 / 4096, -17390 / 4096, -543 / 4096 ]
])

RGB2ITP = LMS2ITP_Mat * RGB2LMS_Mat * Matrix([r, g, b])
ITP2RGB = RGB2LMS_Mat.inv() * LMS2ITP_Mat.inv() * Matrix([I, T, P])
ITPHue = atan2(RGB2ITP[1], RGB2ITP[2])

ITPHuePartial = diff(ITPHue, r, g, b)

#T値の偏微分
ITP_T_RedPartial = diff(RGB2ITP[1], r) #0.152
ITP_T_GrePartial = diff(RGB2ITP[1], g) #-1.41
ITP_T_BluPartial = diff(RGB2ITP[1], b) #1.26
print("CT Partial. Tred=%f, Tblue=%f, Tgreen= %f" % (ITP_T_RedPartial, ITP_T_GrePartial, ITP_T_BluPartial))

#P値の偏微分
ITP_P_RedPartial = diff(RGB2ITP[2], r) #1.09
ITP_P_GrePartial = diff(RGB2ITP[2], g) #-0.775
ITP_P_BluPartial = diff(RGB2ITP[2], b) #-0.318
print("CP Partial. Pred=%f, Pblue=%f, Pgreen= %f" % (ITP_P_RedPartial, ITP_P_GrePartial, ITP_P_BluPartial))

#色相関数の偏微分
HRedPartial = diff(ITPHue, r)
HGrePartial = diff(ITPHue, g)
HBluPartial = diff(ITPHue, b)

#ITP色相の関数化
ITPHue = lambdify((r_, g_, b_), ITPHue.subs([(r, r_), (g, g_), (b, b_)]), "numpy")
ITPHuePartial = lambdify((r_, g_, b_), ITPHuePartial.subs([(r, r_), (g, g_), (b, b_)]), "numpy")

#偏微分をlambdifyで関数化
HRedPartial = lambdify((r_, g_, b_), HRedPartial.subs([(r, r_), (g, g_), (b, b_)]), "numpy")
HGrePartial = lambdify((r_, g_, b_), HGrePartial.subs([(r, r_), (g, g_), (b, b_)]), "numpy")
HBluPartial = lambdify((r_, g_, b_), HBluPartial.subs([(r, r_), (g, g_), (b, b_)]), "numpy")

#RGB<-->IPT色空間へ変換する関数の関数化
RGB2ITP = lambdify((r_, g_, b_), (RGB2ITP[0].subs([(r, r_), (g, g_), (b, b_)]), RGB2ITP[1].subs([(r, r_), (g, g_), (b, b_)]), RGB2ITP[2].subs([(r, r_), (g, g_), (b, b_)])), "numpy")
ITP2RGB = lambdify((I_, T_, P_), (ITP2RGB[0].subs([(I, I_), (T, T_), (P, P_)]), ITP2RGB[1].subs([(I, I_), (T, T_), (P, P_)]), ITP2RGB[2].subs([(I, I_), (T, T_), (P, P_)])), "numpy")

#rgb --> ITPへ変換する式の整理版
#print(N(radsimp(LMS2ITP_Mat, 3)))


#------------------------------------------------------------------------
inputImgPath = "img/s35_4.jpg"
outputImgPath = "outimg/test_ACE-HueC.jpg"
doSignalConvert = False
doHueCorrection = True

#Pathに日本語が含まれるとエラー
img = cv2.imread(inputImgPath, cv2.IMREAD_COLOR)

#正規化と整形
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB) / 255.
RGB = np.reshape(img, (img.shape[0] * img.shape[1], 3))

#rgbから各色空間へ変換
imgITP = np.asarray(RGB2ITP(RGB[:, 0], RGB[:, 1], RGB[:, 2])).T
imgICH = ITP2ICH(imgITP)

#入力画像の色相値([-pi, pi])
imgHue = ITPHue(RGB[:, 0], RGB[:, 1], RGB[:, 2])

#-----------------------------------------------
omegaFFT, omega = computeOmegaTrans(img.shape[0], img.shape[1])

mappedImg = np.copy(RGB)
myu = np.average(RGB, axis=0)

deltaT = 0.01
gamma = 0.0
alpha = np.abs(gamma) / 20
beta = 1

ratio = 0.5

hueRate = 0.2


for it in range(37):
    print("---------Iter:%2d, Gamma:%f, Alpha:%f---------" % (it, gamma, alpha))

    rgbBefore = np.zeros((img.shape[0] * img.shape[1], 3), dtype='float64')

    newHue = np.random.rand(img.shape[0] * img.shape[1])
    redDiff = RGB[:, 0] 
    greDiff = RGB[:, 1] 
    bluDiff = RGB[:, 2] 
    

    for k in range(10000):
        #エンハンス
        for colorCh in range(3):
            contrast = RIslow(omegaFFT, mappedImg[:, colorCh], img.shape[0], img.shape[1], alpha)
            molecule = mappedImg[:, colorCh] + deltaT * (beta * RGB[:, colorCh] + (0.5 * gamma * contrast))
            mappedImg[:, colorCh] = molecule / (1 + deltaT * beta)

        #二乗平均平方根誤差(RMSE)
        enhanceLoss = np.sqrt(mean_squared_error(mappedImg, rgbBefore))

        #色相修正
        if doHueCorrection:
            #現在のICH色情報を計算
            ITPNow = np.asarray(RGB2ITP(mappedImg[:, 0], mappedImg[:, 1], mappedImg[:, 2]), dtype='float64').T
            HueNow = ITPHue(mappedImg[:, 0], mappedImg[:, 1], mappedImg[:, 2])

            denominator = ITPNow[:, 1]**2 + ITPNow[:, 2]**2
            hueDiff = HueNow - imgHue

            redDiff = redDiff - 2 * hueRate * hueDiff * (( 0.152188 * ITPNow[:, 2] - 1.093136 * ITPNow[:, 1]) / denominator)
            greDiff = greDiff - 2 * hueRate * hueDiff * ((-1.419960 * ITPNow[:, 2] + 0.774947 * ITPNow[:, 1]) / denominator)
            bluDiff = bluDiff - 2 * hueRate * hueDiff * (( 1.267772 * ITPNow[:, 2] + 0.318189 * ITPNow[:, 1]) / denominator)

            mappedImg = np.c_[redDiff, greDiff, bluDiff]
            
            newHue = ITPHue(mappedImg[:, 0], mappedImg[:, 1], mappedImg[:, 2])

            #ITPBefore = np.asarray(RGB2ITP(rgbBefore[:, 0], rgbBefore[:, 1], rgbBefore[:, 2])).T
            #ICHBefore = ITP2ICH(ITPBefore)

            #ITP値の微分値を計算
            #ITPDiff = ITPBefore - ITPNow

            #TP(RGBからなる関数)の変化分を計算
            #differential = (ITPDiff[:, 1] * ITPNow[:, 2] - ITPDiff[:, 2] * ITPNow[:, 1]) / (ITPNow[:, 1]**2 + ITPNow[:, 2]**2)
            #differential = 1 / (1 + (ITPNow[:, 1] / ITPNow[:, 2])**2)
            #differential = ICHNow[:, 2] - ICHBefore[:, 2]
            #differential = np.asarray(ITPHuePartial(mappedImg[:, 0], mappedImg[:, 1], mappedImg[:, 2])).T
            #print(differential[0:5])
            
            #色相計算の更新
            #newHue = newHue - (1 - ratio) * 0.1 * (ICHNow[:, 2] - imgHue)
            #newICH = np.c_[ICHNow[:, 0], ICHNow[:, 1], newHue]
            
            #ICH色空間をRGB色空間に戻す
            #ITP = ICH2ITP(newICH)
            #mappedImg = np.asarray(ITP2RGB(ITP[:, 0], ITP[:, 1], ITP[:, 2])).T

            #二乗平均平方根誤差(RMSE)による入力画像の色相差の計算
            hueLoss = np.sqrt(mean_squared_error(newHue, imgHue))
            #print("delta H=%f, Diff=%f" % (hueLoss, np.abs(np.sum(newHue - imgHue))))

        if doHueCorrection:
            allLoss = ratio * enhanceLoss + (1 - ratio) * hueLoss
            print("Iter:%2d, k:%d, All Loss=%f, Enhance Loss=%f, Hue Loss=%f" % (it, k, allLoss, enhanceLoss, hueLoss))

            if allLoss< 1e-2:
                break
            else:
                rgbBefore = copy.deepcopy(mappedImg)
        else:
            if enhanceLoss < 1e-4:
                print("Iter:%2d, Enhance Loss=%f" % (k, enhanceLoss))
                break
            else:
                rgbBefore = np.copy(mappedImg)
    
    gamma += 0.01
    alpha += np.abs(gamma) / 20

#入力画像との色相の絶対平均誤差の計算
ICH = ITP2ICH(np.asarray(RGB2ITP(mappedImg[:, 0], mappedImg[:, 1], mappedImg[:, 2])).T)
print("Hue Loss : %f" % (np.sqrt(mean_squared_error(ICH[:, 2], imgHue))))

#視覚実験に基づくLED制御値への変換
if doSignalConvert:
    Ml = np.matrix([[0.7126, 0.1142, 0.0827],
                    [0.0236, 0.3976, 0.0256],
                    [0.0217, 0.0453, 0.5512]], dtype=np.float)
    
    signal = [np.dot(Ml, rgb) for rgb in mappedImg]
    mappedImg = np.array(signal)

#(H,W,3)に整形とクリップ処理
mappedImg = np.clip(mappedImg.reshape((img.shape[0], img.shape[1], 3)), 0, 1)

'''
mappedxy = colour.XYZ_to_xy(colour.sRGB_to_XYZ(mappedImg))
imagexy = colour.XYZ_to_xy(colour.sRGB_to_XYZ(RGB))
#-----------------------------------------------
#データのプロット

#色域、ホワイトポイントのプロット
sRGBGamutBoundary = np.array([
    [0.6400, 0.3300],
    [0.3000, 0.6000],
    [0.1500, 0.0600],
    [0.6400, 0.3300]])

fig = plt.figure()
fig.suptitle('r-image of sRGB and LEDRGB', fontsize=14, fontweight='bold')

ax1 = fig.add_subplot(221)
colour.plotting.chromaticity_diagram_plot_CIE1931(bounding_box=(-0.1, 0.9, -0.1, 0.9), standalone=False)
ax1.plot(sRGBGamutBoundary[:, 0], sRGBGamutBoundary[:, 1], color="black", linestyle='dotted')
ax1.plot(sRGBGamutBoundary[:, 0], sRGBGamutBoundary[:, 1], 's', color="black", markersize=6, label="sRGB Gamut Boundary Point")
ax1.plot(0.3127, 0.3290, 'v', color="black", markersize=4, label="sRGB WhitePoint")
ax1.plot(imagexy[:, 0], imagexy[:, 1],  's', markersize=2, label="sRGB Pixel")
ax1.legend()

ax2 = fig.add_subplot(222)
colour.plotting.chromaticity_diagram_plot_CIE1931(bounding_box=(-0.1, 0.9, -0.1, 0.9), standalone=False)
ax2.plot(sRGBGamutBoundary[:, 0], sRGBGamutBoundary[:, 1], color="black", linestyle='dotted')
ax2.plot(sRGBGamutBoundary[:, 0], sRGBGamutBoundary[:, 1], 's', color="black", markersize=6, label="sRGB Gamut Boundary Point")
ax2.plot(0.3127, 0.3290, 'v', color="black", markersize=4, label="sRGB WhitePoint")
ax2.plot(mappedxy[:, 0], mappedxy[:, 1],  's', markersize=2, label="Mapped Pixel")
ax2.legend()

ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)

ax3.set_title("Original Image")
ax4.set_title("Mapped Image")

ax3.imshow(img, interpolation='none')
ax4.imshow(mappedImg, interpolation='none')

plt.show()
'''
im = Image.fromarray(np.uint8(mappedImg.reshape((img.shape[0], img.shape[1], 3)) * 255))
im.save(outputImgPath)
