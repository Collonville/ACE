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

#1列目:LED制御値、2~4列目:XYZ値
LEDAchromaticXYZ = np.loadtxt("LEDAchromatic.csv", delimiter=",", skiprows=1)
LEDxy = colour.XYZ_to_xy(LEDAchromaticXYZ[:, 1:4])

#無彩色RGB値の生成
achromaticLine = np.linspace(0.0, 1.0, 18, endpoint=True, dtype='float64')
achromaticRGB = np.c_[achromaticLine, achromaticLine, achromaticLine]

XYZ = sRGB_to_XYZ(achromaticRGB)
xy = XYZ_to_xy(XYZ)

print(xy)

#-----------------------------------------------
#データのプロット

fig = plt.figure()
fig.suptitle('r-image of sRGB and LEDRGB', fontsize=14, fontweight='bold')

ax1 = fig.add_subplot(111)
colour.plotting.CIE_1931_chromaticity_diagram_plot(bounding_box=(-0.1, 0.9, -0.1, 0.9), standalone=False)
ax1.plot(LEDxy[:, 0], LEDxy[:, 1], 'o', color="red", markersize=3)
ax1.plot(xy[:, 0], xy[:, 1], 'o', color="green", markersize=5)
ax1.plot(0.3127, 0.3290, 'v', color="black", markersize=4, label="sRGB WhitePoint")
ax1.legend()

plt.show()


