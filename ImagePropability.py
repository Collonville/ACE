import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import sklearn
import win_unicode_console

import ImageFeature

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=10, suppress=True, threshold=np.inf, linewidth=100)

def getImageRGBFromPath(filePath):
    inputImg = cv2.imread(filePath, cv2.IMREAD_COLOR)

    #正規化と整形
    inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
    rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))

    return rgb, inputImg.shape[0], inputImg.shape[1]

#------------------------------------------------------------------------
#読み込みたい画像集合のパス(例:outimg/ACE2/strawberry)
inputImgPath = sys.argv[1]

#学習済みパラメータの取得
intercept = np.load("LogisticRegresion/intercept.npy")
coef = np.load("LogisticRegresion/coef.npy")

#標準器の読み込み
scaler = sklearn.externals.joblib.load("LogisticRegresion/FeatureScaler.pkl")

#画像特徴量取得に関するインスタンス
imageFeature = ImageFeature.ImageFeature()

features = np.empty((0, coef.shape[0]))

#initRGB = getImageRGBFromPath("img/All/" + fileName + ".jpg")
initRGB = []
for it in range(100):
    rgb, imgH, imgW = getImageRGBFromPath(inputImgPath + "_" + str(it) + ".jpg")

    #特徴量の取得
    feature = imageFeature.getImageFeatureFromRGB(rgb, imgH, imgW, initRGB)
    feature[np.isnan(feature)] = 0

    features = np.r_[features, feature]

#特徴量の標準化
features = scaler.transform(features)

#選択確率の計算
portion = intercept + np.dot(coef, features.T)
propability = 1. / (1. + np.exp(-portion))

#合計を1に正規化
propability = propability / np.sum(propability)

print("Max iter=%d, propability=%f" % (np.argmax(propability), np.max(propability)))

#補正値を加えた後の最良推定反復回数
print("Correction value=" + str(np.argmax(propability) + 8))

plt.plot(propability)
plt.plot(np.argmax(propability), np.max(propability), marker='o', color='r')
plt.text(np.argmax(propability) + 5, np.max(propability), "Iter=%d, Propability=%f" % (np.argmax(propability), np.max(propability)))
plt.xlabel("Iter")
plt.ylabel("Propability")
plt.grid(True)
plt.show()
