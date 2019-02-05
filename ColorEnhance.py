import sys

import colour
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import win_unicode_console
from colour.models import *
from colour.plotting import *

import ACE

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=10, suppress=True, threshold=np.inf, linewidth=100)
'''
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

@numba.jit('f8[:](f8[:], f8[:], f8, f8[:], f8, f8, i4, i4)')
def energy(I, I0, alpha, myu, beta, gamma, width, height):
    first = 0
    second = 0
    third = 0

    for colorCh in range(3):
        #1項目：灰色仮説
        first += alpha * 0.5 * np.sum((I[:, colorCh] - myu[colorCh]) ** 2)

        #2項目：入力画像との色差
        second += beta * 0.5 * np.sum((I[:, colorCh] - I0[:, colorCh]) ** 2)

        #3項目：ローカルコントラスト
        for y in range(height):
            for x in range(width):
                third += omega[y * width + x] * np.abs(I[x, colorCh] - I[y, colorCh])
    
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

    return 0.5 * np.sum((ITPHue(r, g, b) - imgHue)**2)

def optimaizeFuncGradient(rgb):
    r, g, b = np.split(rgb, 3)

    hueDiff = ITPHue(r, g, b) - imgHue

    rPrime = hueDiff * np.asarray(HRedPartial(r, g, b), dtype='float64').T
    gPrime = hueDiff * np.asarray(HGrePartial(r, g, b), dtype='float64').T
    bPrime = hueDiff * np.asarray(HBluPartial(r, g, b), dtype='float64').T

    return np.r_[rPrime, gPrime, bPrime]

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

#------------------------------------------------------------------------
fileName = "strawberry"
inputImgPath = "img/All/" + fileName + ".jpg"
#outputImgPath = "outimg/EnhacedImage/WithHue/" + fileName
outputImgPath = "outimg/ACE2/"
doHueCorrection = False     #色相補正
OUT_CONSECUTIVE_IMG = True #連続画像作成
OUT_SIGNAL_IMG = False

#Pathに日本語が含まれるとエラー
inputImg = cv2.imread(inputImgPath, cv2.IMREAD_COLOR)

#正規化と整形
inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
RGB = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))

#rgbから各色空間へ変換
imgITP = np.asarray(RGB2ITP(RGB[:, 0], RGB[:, 1], RGB[:, 2])).T
imgICH = ITP2ICH(imgITP)

#入力画像の色相値([-pi, pi])
imgHue = ITPHue(RGB[:, 0], RGB[:, 1], RGB[:, 2])

#------------------------------------------------------------------------
omegaFFT, omega = computeOmegaTrans(inputImg.shape[0], inputImg.shape[1])

mappedImg = np.copy(RGB)
myu = np.mean(RGB, axis=0)

deltaT = 0.1
gamma = 0.0
alpha = 0.0
beta = 1

ratio = 0.5
lossBefore = 0

MAX_ITER = 100
feature = np.zeros((MAX_ITER, 4))
momentFeature = np.zeros((MAX_ITER, 12))

#エンハンスエネルギー
energySet = np.empty((0, 4))
hueLossSet = np.empty(0)

#ACEと色相保持を交互にする方法

for it in range(MAX_ITER):
    print("---------Iter:%2d, Gamma:%f, Alpha:%f---------" % (it, gamma, alpha))

    rgbBefore = np.zeros((inputImg.shape[0] * inputImg.shape[1], 3))
    
    for k in range(500):
        #エンハンス
        for colorCh in range(3):
            contrast = RIslow(omegaFFT, mappedImg[:, colorCh], inputImg.shape[0], inputImg.shape[1], 1)
            molecule = mappedImg[:, colorCh] + deltaT * (alpha * myu[colorCh] + beta * RGB[:, colorCh] + (0.5 * gamma * contrast))
            mappedImg[:, colorCh] = molecule / (1 + deltaT * (alpha + beta))

        #二乗平均平方根誤差(RMSE)
        enhanceLoss = np.sqrt(mean_squared_error(mappedImg, rgbBefore))

        #色相修正
        if doHueCorrection:
            #最適化
            hueOpt = minimize(optimaizeFunc, mappedImg.flatten('F'), 
                                method='CG',  
                                options={'gtol': 1e-5, 'disp': False},
                                jac=optimaizeFuncGradient
                            )

            #最適化結果を整形
            r, g, b = np.split(hueOpt.x, 3)
            mappedImg = np.c_[r, g, b]

            #入力画像との色相差を計算
            hueLoss = np.sqrt(mean_squared_error(ITPHue(mappedImg[:, 0], mappedImg[:, 1], mappedImg[:, 2]), imgHue))
        
        energySet = np.r_[energySet, energy(mappedImg, RGB, alpha, myu, beta, gamma, inputImg.shape[0], inputImg.shape[1])]

        if doHueCorrection:
            allLoss = ratio * enhanceLoss + (1 - ratio) * hueLoss
            loss = np.abs(allLoss - lossBefore)
            print("Iter:%2d, k:%4d, All Loss=%f, Enhance Loss=%f, Hue Loss=%f" % (it, k, loss, enhanceLoss, hueLoss))
            
            if loss < 1e-4:
                break
            else:
                lossBefore = allLoss
                rgbBefore = copy.deepcopy(mappedImg)
        else:
            if enhanceLoss < 1e-4:
                print("Iter:%2d, Enhance Loss=%f" % (k, enhanceLoss))
                break
            else:
                rgbBefore = copy.deepcopy(mappedImg)

    if OUT_CONSECUTIVE_IMG:
        img_ = np.clip(mappedImg, 0, 1)
        im = Image.fromarray(np.uint8(img_.reshape((inputImg.shape[0], inputImg.shape[1], 3)) * 255))
        im.save(outputImgPath + fileName + "_" + str(it) + ".jpg", quality=100)
    
    if OUT_SIGNAL_IMG:
        Ml = np.matrix([[0.7126, 0.1142, 0.0827],
                    [0.0236, 0.3976, 0.0256],
                    [0.0217, 0.0453, 0.5512]], dtype=np.float)
    
        signal = [np.dot(Ml, rgb) for rgb in mappedImg]
        signalImg = np.array(signal)

        img_ = np.clip(signalImg, 0, 1)
        im = Image.fromarray(np.uint8(img_.reshape((inputImg.shape[0], inputImg.shape[1], 3)) * 255))
        im.save("outimg/SignalImage/WithHue/" + fileName + "_" + str(it) + "_Signal.jpg", quality=100)

    gamma += 0.01
    alpha += np.abs(gamma) / 20


#入力画像との色相の二乗平均平方根誤差(RMSE)
hueLoss = np.sqrt(mean_squared_error(ITPHue(mappedImg[:, 0], mappedImg[:, 1], mappedImg[:, 2]), imgHue))
print("Hue Loss : %f" % hueLoss)
'''

fileName = sys.argv[1]
energySet, hueLossSet = ACE.doEnhanceMethod2(fileName)

#------------------------------------------------------------------------
fig = plt.figure()
fig.suptitle(fileName + ", With hue correction", fontsize=12, fontweight='bold')

ax1 = fig.add_subplot(131)
ax1.plot(energySet[:, 0], label="All Energy")
ax1.plot(energySet[:, 1], label="First(Gray world)")
ax1.plot(energySet[:, 2], label="Second(Diff from init)")
ax1.legend(loc='upper left')
ax1.grid()

ax2 = fig.add_subplot(132)
ax2.plot(energySet[:, 3], label="Third(Local contrast)")
ax2.legend(loc='upper left')
ax2.grid()

ax3 = fig.add_subplot(133)
ax3.plot(hueLossSet, label="Hue loss(RMSE)")
ax3.legend(loc='upper left')
ax3.grid()

plt.show()