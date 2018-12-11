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

#------------------------------------------------------------------------
fileName = sys.argv[1]
inputImgPath = "outimg/continuity_hue/All/" + fileName

#学習済みパラメータの取得
intercept = np.load("LogisticRegresion/intercept.npy")
coef = np.load("LogisticRegresion/coef.npy")

#標準器の読み込み
scaler = sklearn.externals.joblib.load("LogisticRegresion/FeatureScaler.pkl")

imageFeature = ImageFeature.ImageFeature()

features = np.empty((0, coef.shape[0]))

for it in range(100):
    inputImg = cv2.imread(inputImgPath + "_" + str(it) + ".jpg", cv2.IMREAD_COLOR)

    #正規化と整形
    inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
    rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))

    #特徴量の計算
    feature = imageFeature.getImageFeatureFromRGB(rgb)
    feature[np.isnan(feature)] = 0

    features = np.r_[features, feature]

#特徴量の標準化
features = scaler.transform(features)

#選択確率の計算
portion = intercept + np.dot(coef, features.T)
propability = 1. / (1. + np.exp(-portion))

print("Max iter=%d, propability=%f" % (np.argmax(propability), np.max(propability)))
plt.plot(propability)
plt.plot(np.argmax(propability), np.max(propability), marker='o', color='r')
plt.text(np.argmax(propability) + 5, np.max(propability), "Iter=%d, Propability=%f" % (np.argmax(propability), np.max(propability)))
plt.xlabel("Iter")
plt.ylabel("Propability")
plt.grid(True)
plt.show()
