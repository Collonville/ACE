import copy
import glob
import itertools
import math
import sys

import colour
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

import cv2

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=10, suppress=True, threshold=np.inf, linewidth=100)









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

im = Image.fromarray(np.uint8(mappedImg.reshape((img.shape[0], img.shape[1], 3)) * 255))
im.save(outputImgPath)
'''
