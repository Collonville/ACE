import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import win_unicode_console

import ImageFeature

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=10, suppress=True, threshold=np.inf, linewidth=100)

#------------------------------------------------------------------------
fileName = sys.argv[1]

imageFeature = ImageFeature.ImageFeature()

inputImg = cv2.imread(fileName, cv2.IMREAD_COLOR)

#正規化と整形
inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))

#特徴量の計算
feature = imageFeature.getImageFeatureFromRGB(rgb)

print(feature)


