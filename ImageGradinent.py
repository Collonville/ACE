import copy
import itertools
import math
import sys

import numpy as np

import colour
import cv2
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import win_unicode_console
from colour.models import *
from colour.plotting import *
from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageColor
from scipy.stats import entropy

from skimage import filters

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=8, suppress=True, threshold=np.inf, linewidth=100)

argv = sys.argv

#------------------------------------------------------------------------
ImgPath1 = argv[1]
ImgPath2 = argv[2]

#Pathに日本語が含まれるとエラー
img1 = cv2.imread(ImgPath1, cv2.IMREAD_COLOR)
img2 = cv2.imread(ImgPath2, cv2.IMREAD_COLOR)

#正規化と整形
img1 = cv2.cvtColor(img1, cv2.COLOR_BGR2RGB)
img2 = cv2.cvtColor(img2, cv2.COLOR_BGR2RGB)
RGB1 = np.reshape(img1, (img1.shape[0] * img1.shape[1], 3)) / 255.0
RGB2 = np.reshape(img2, (img2.shape[0] * img2.shape[1], 3)) / 255.0

#------------------------------------------------------------------------
BIN_NUM = 100

'''
hsv1 = colour.RGB_to_HSL(RGB1)
hsv2 = colour.RGB_to_HSL(RGB2)

fig, axes = plt.subplots(nrows=1, ncols=2)
ax = axes.ravel()

hist, bins, pathes = ax[0].hist(hsv1[:, 1].flatten(), bins=BIN_NUM)

print(entropy(hist, base=2))
ax[0].set_title('Original Image')

hist, bins, pathes = ax[1].hist(hsv2[:, 1].flatten(), bins=BIN_NUM)
print(entropy(hist, base=2))
ax[1].set_title('Original Image')
plt.show()
'''



#-----------------------------------------------
redGrad1 = filters.sobel(img1[:, :, 0])
greGrad1 = filters.sobel(img1[:, :, 1])
bluGrad1 = filters.sobel(img1[:, :, 2])
imgGrad1 = np.sqrt(redGrad1**2 + greGrad1**2 + bluGrad1**2)
aveGrad1 = np.sum(imgGrad1) / (img1.shape[0] * img1.shape[1])

redGrad2 = filters.sobel(img2[:, :, 0])
greGrad2 = filters.sobel(img2[:, :, 1])
bluGrad2 = filters.sobel(img2[:, :, 2])
imgGrad2 = np.sqrt(redGrad2**2 + greGrad2**2 + bluGrad2**2)
aveGrad2 = np.sum(imgGrad2) / (img2.shape[0] * img2.shape[1])

print("Average Gradient:%f, %f" % (aveGrad1, aveGrad2))

#-----------------------------------------------
#勾配情報のSurfaceプロット
x = np.arange(0, img1.shape[0])
y = np.arange(0, img1.shape[1])
X, Y = np.meshgrid(x, y)

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.invert_xaxis()
surf = ax.plot_surface(X, Y, imgGrad1, cmap=cm.coolwarm)
#-----------------------------------------------

#勾配値の最大値を取得
maxValue = np.max(np.maximum(imgGrad1, imgGrad2))

fig, axes = plt.subplots(nrows=2, ncols=5, sharex=True, sharey=True, figsize=(10, 10))
ax = axes.ravel()

ax[0].imshow(np.asarray(img1))
ax[0].set_title('Original Image')
ax[1].imshow(imgGrad1, cmap='gray', vmax=maxValue)
ax[1].set_title('Image Gradienat')
ax[2].imshow(redGrad1, cmap='gray', vmax=maxValue)
ax[2].set_title('Red Gradienat')
ax[3].imshow(greGrad1, cmap='gray', vmax=maxValue)
ax[3].set_title('Green Gradienat')
ax[4].imshow(bluGrad1, cmap='gray', vmax=maxValue)
ax[4].set_title('Blue Gradienat')

ax[5].imshow(np.asarray(img2))
ax[5].set_title('Original Image')
ax[6].imshow(imgGrad2, cmap='gray', vmax=maxValue)
ax[6].set_title('Image Gradienat')
ax[7].imshow(redGrad2, cmap='gray', vmax=maxValue)
ax[7].set_title('Red Gradienat')
ax[8].imshow(greGrad2, cmap='gray', vmax=maxValue)
ax[8].set_title('Green Gradienat')
ax[9].imshow(bluGrad2, cmap='gray', vmax=maxValue)
ax[9].set_title('Blue Gradienat')

for a in ax:
    a.axis('off')

plt.tight_layout()
plt.show()
