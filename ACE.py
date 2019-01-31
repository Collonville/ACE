import copy
import glob
import itertools
import math
import sys

import colour
import cv2
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

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=10, suppress=True, threshold=np.inf, linewidth=100)
 
@numba.jit('f8[:](f8[:], f8[:], i8, i8, f4)')
def RIslow(omega, enhancedImg, imgH, imgW, slope):
    valsum = np.zeros(imgW * imgH)

    for y in range(imgW * imgH):
        sums = 0
        for x in range(imgW * imgH):
            sums += omega[x] * np.minimum(np.maximum(slope * (enhancedImg[x] - enhancedImg[y]), -1), 1)

        valsum[y] = sums
    
    return valsum
'''
def RI(self, img):
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
'''
def computeOmegaTrans(imgH, imgW):
    omegaDistanceFunc = "Euclidean"

    omega = np.zeros(imgW * imgH)

    if omegaDistanceFunc is "Gaussian":
        sigma = 10
        for y in range(imgH):
            for x in range(imgW):
                omega[y * imgW + x] += np.exp(-(x**2 + y**2) / (2 * sigma**2))
    elif omegaDistanceFunc is "Euclidean":
        for y in range(imgH):
            for x in range(imgW):
                omega[y * imgW + x] = 0 if (x == 0 & y == 0) else 1. / np.sqrt(x**2 + y**2)

    #omegaの合計値が1になるように正規化
    omega = omega / np.sum(omega)

    return rfft(omega), omega

@numba.jit('f8[:](f8[:], f8[:], i8 , i8, f8[:], f8[:], f8, f8, f8)')
def imageEnergy(enhancedImg, Img0, imgH, imgW, omega, myu, alpha, beta, gamma):
    first = 0
    second = 0
    third = 0

    for colorCh in range(3):
        #1項目：灰色仮説
        first += np.sum((enhancedImg[:, colorCh] - myu[colorCh]) ** 2)

        #2項目：入力画像との色差
        second += np.sum((enhancedImg[:, colorCh] - Img0[:, colorCh]) ** 2)

        #3項目：ローカルコントラスト
        for y in range(imgH):
            for x in range(imgW):
                third += omega[y * imgW + x] * np.abs(enhancedImg[x, colorCh] - enhancedImg[y, colorCh])
    
    first = alpha * 0.5 * first
    second = beta * 0.5 * second
    third = gamma * 0.5 * third

    #全エネルギーの合計
    allEnergy = first + second - third

    return np.c_[allEnergy, first, second, third]

def ITP2ICH(ITP):
    C = np.sqrt(ITP[:, 1]**2 + ITP[:, 2]**2)
    H = np.arctan2(ITP[:, 1], ITP[:, 2])
    
    return np.c_[ITP[:, 0], C, H]

def ICH2ITP(ICH):
    P = ICH[:, 1] * np.cos(ICH[:, 2])
    T = ICH[:, 1] * np.sin(ICH[:, 2])

    return np.c_[ICH[:, 0], T, P]

def optimaizeFunc(rgb):
    r, g, b = np.split(rgb, 3)

    return 0.5 * np.sum((ITPHue(r, g, b) - ITPHue0)**2)

def optimaizeFuncGradient(rgb):
    r, g, b = np.split(rgb, 3)

    hueDiff = ITPHue(r, g, b) - ITPHue0

    rPrime = hueDiff * np.asarray(HRedPartial(r, g, b), dtype='float64').T
    gPrime = hueDiff * np.asarray(HGrePartial(r, g, b), dtype='float64').T
    bPrime = hueDiff * np.asarray(HBluPartial(r, g, b), dtype='float64').T

    return np.r_[rPrime, gPrime, bPrime]

def getITPPartial():
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

    #T値の偏微分
    ITP_T_RedPartial = diff(RGB2ITP[1], r) # 0.152
    ITP_T_GrePartial = diff(RGB2ITP[1], g) #-1.41
    ITP_T_BluPartial = diff(RGB2ITP[1], b) # 1.26
    print("CT Partial. Tred=%f, Tblue=%f, Tgreen= %f" % (ITP_T_RedPartial, ITP_T_GrePartial, ITP_T_BluPartial))

    #P値の偏微分
    ITP_P_RedPartial = diff(RGB2ITP[2], r) # 1.09
    ITP_P_GrePartial = diff(RGB2ITP[2], g) #-0.775
    ITP_P_BluPartial = diff(RGB2ITP[2], b) #-0.318
    print("CP Partial. Pred=%f, Pblue=%f, Pgreen= %f" % (ITP_P_RedPartial, ITP_P_GrePartial, ITP_P_BluPartial))

    #色相関数の偏微分
    HRedPartial = diff(ITPHue, r)
    HGrePartial = diff(ITPHue, g)
    HBluPartial = diff(ITPHue, b)

    #ITP色相の関数化
    ITPHue = lambdify((r_, g_, b_), ITPHue.subs([(r, r_), (g, g_), (b, b_)]), "numpy")

    #偏微分をlambdifyで関数化
    HRedPartial = lambdify((r_, g_, b_), HRedPartial.subs([(r, r_), (g, g_), (b, b_)]), "numpy")
    HGrePartial = lambdify((r_, g_, b_), HGrePartial.subs([(r, r_), (g, g_), (b, b_)]), "numpy")
    HBluPartial = lambdify((r_, g_, b_), HBluPartial.subs([(r, r_), (g, g_), (b, b_)]), "numpy")

    #RGB<-->IPT色空間へ変換する関数の関数化
    RGB2ITP = lambdify((r_, g_, b_), (RGB2ITP[0].subs([(r, r_), (g, g_), (b, b_)]), RGB2ITP[1].subs([(r, r_), (g, g_), (b, b_)]), RGB2ITP[2].subs([(r, r_), (g, g_), (b, b_)])), "numpy")
    ITP2RGB = lambdify((I_, T_, P_), (ITP2RGB[0].subs([(I, I_), (T, T_), (P, P_)]), ITP2RGB[1].subs([(I, I_), (T, T_), (P, P_)]), ITP2RGB[2].subs([(I, I_), (T, T_), (P, P_)])), "numpy")

    #rgb --> ITPへ変換する式の整理版
    #print(N(radsimp(LMS2ITP_Mat, 3)))
    return ITPHue, HRedPartial, HGrePartial, HBluPartial

def readImage(inputImgPath, ITPHue):
    #Pathに日本語が含まれるとエラー
    inputImg = cv2.imread(inputImgPath, cv2.IMREAD_COLOR)

    #正規化と整形
    inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
    rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))

    myu = np.mean(rgb, axis=0)

    '''
    #rgbから各色空間へ変換
    imgITP = np.asarray(RGB2ITP(RGB[:, 0], RGB[:, 1], RGB[:, 2])).T
    imgICH = ITP2ICH(imgITP)
    '''

    #入力画像の色相値([-pi, pi])
    imgHue = ITPHue(rgb[:, 0], rgb[:, 1], rgb[:, 2])

    return rgb, myu, imgHue, inputImg.shape[0], inputImg.shape[1]

def doEnhanceMethod1(fileName_):    
    MAX_ITER = 100
    DO_HUE_CORRECTION      = False#色相補正
    OUTPUT_CONSECUTIVE_IMG = True#連続画像作成
    OUTPUT_SIGNAL_IMG      = False#制御値画像作成
 
    #ACEの計算に使う定数値
    deltaT = 0.01
    alpha = 0.0
    beta = 1.0
    gamma = 0.0
    slope = 1.0
 
    #入力画像、出力画像のパス
    fileName = fileName_
    inputImgPath = "img/All/" + fileName_ + ".jpg"
    enhanceImgOutputPath = "outimg/ACE2/"
    signalImgOutputPath = "outimg/ACE2/"
 
    #色相計算の微分関数
    ITPHue, HRedPartial, HGrePartial, HBluPartial = getITPPartial()

    #エンハンス対象の画像情報
    Img0, myu, ITPHue0, imgW, imgH = readImage(inputImgPath, ITPHue)
    enhancedImg = np.copy(Img0)

    #コントラストカーネル
    omegaFFT, omega = computeOmegaTrans(imgH, imgW)

    ratio = 0.5
    lossBefore = 0

    #エンハンスエネルギー
    energySet = np.empty((0, 4))
    hueLossSet = np.empty(0)

    for it in range(MAX_ITER):
        print("---------Iter:%2d, Gamma:%f, Alpha:%f---------" % (it, gamma, alpha))

        rgbBefore = np.zeros((imgH * imgW, 3))
        
        for k in range(500):
            #エンハンス
            for colorCh in range(3):
                contrast = RIslow(omegaFFT, enhancedImg[:, colorCh], imgH, imgW, slope)
                molecule = enhancedImg[:, colorCh] + deltaT * (alpha * myu[colorCh] + beta * Img0[:, colorCh] + (0.5 * gamma * contrast))
                enhancedImg[:, colorCh] = molecule / (1 + deltaT * (alpha + beta))

            #二乗平均平方根誤差(RMSE)
            enhanceLoss = np.sqrt(mean_squared_error(enhancedImg, rgbBefore))

            #色相修正
            if DO_HUE_CORRECTION:
                #最適化
                hueOpt = minimize(optimaizeFunc, enhancedImg.flatten('F'), 
                                    method='CG',  
                                    options={'gtol': 1e-5, 'disp': False},
                                    jac=optimaizeFuncGradient
                                )

                #最適化結果を整形
                r, g, b = np.split(hueOpt.x, 3)
                enhancedImg = np.c_[r, g, b]

                #入力画像との色相差を計算
                hueLoss = np.sqrt(mean_squared_error(ITPHue(enhancedImg[:, 0], enhancedImg[:, 1], enhancedImg[:, 2]), ITPHue0))
            
            energySet = np.r_[energySet, imageEnergy(enhancedImg, Img0, imgH, imgW, omega, myu, alpha, beta, gamma)]

            if DO_HUE_CORRECTION:
                allLoss = ratio * enhanceLoss + (1 - ratio) * hueLoss
                loss = np.abs(allLoss - lossBefore)
                print("Iter:%2d, k:%4d, All Loss=%f, Enhance Loss=%f, Hue Loss=%f" % (it, k, loss, enhanceLoss, hueLoss))
                
                if loss < 1e-4:
                    break
                else:
                    lossBefore = allLoss
                    rgbBefore = copy.deepcopy(enhancedImg)
            else:
                if enhanceLoss < 1e-4:
                    print("Iter:%2d, Enhance Loss=%f" % (k, enhanceLoss))
                    break
                else:
                    rgbBefore = copy.deepcopy(enhancedImg)

        if OUTPUT_CONSECUTIVE_IMG:
            img_ = np.clip(enhancedImg, 0, 1)
            im = Image.fromarray(np.uint8(img_.reshape((imgH, imgW, 3)) * 255))
            im.save(enhanceImgOutputPath + fileName + "_" + str(it) + ".jpg", quality=100)
        
        if OUTPUT_SIGNAL_IMG:
            Ml = np.matrix([[0.7126, 0.1142, 0.0827],
                            [0.0236, 0.3976, 0.0256],
                            [0.0217, 0.0453, 0.5512]], dtype=np.float)
        
            signal = [np.dot(Ml, rgb) for rgb in enhancedImg]
            signalImg = np.array(signal)

            img_ = np.clip(signalImg, 0, 1)
            im = Image.fromarray(np.uint8(img_.reshape((imgH * imgW, 3)) * 255))
            im.save(signalImgOutputPath + fileName + "_" + str(it) + "_Signal.jpg", quality=100)

        gamma += 0.01
        alpha += np.abs(gamma) / 20

    #入力画像との色相の二乗平均平方根誤差(RMSE)
    hueLoss = np.sqrt(mean_squared_error(ITPHue(enhancedImg[:, 0], enhancedImg[:, 1], enhancedImg[:, 2]), ITPHue0))
    print("Hue Loss : %f" % hueLoss)

    return energySet, hueLossSet
