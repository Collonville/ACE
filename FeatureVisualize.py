import copy
import itertools
import sys

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np
import win_unicode_console

from mpl_toolkits.mplot3d import Axes3D
from PIL import Image, ImageColor
import cv2

win_unicode_console.enable()

#------------------------------------------------------------------------
inputImgPath = "img/strawberry.jpg"
outputImgPath = "outimg/continuity/strawberry"

fig, axes = plt.subplots(nrows=1, ncols=6)
ax = axes.ravel()

ax[0].plot(feature[:, 0], label="Average Gradient")
ax[0].grid(True)
ax[0].legend()
ax[0].set_xlabel("Iter")

ax[1].plot(feature[:, 1], label="Hue Entropy")
ax[1].plot(feature[:, 2], label="Saturation Entropy")
ax[1].plot(feature[:, 3], label="Luminance Entropy")
ax[1].set_xlabel("Iter")
ax[1].grid(True)
ax[1].legend(loc="upper right")

ax[2].plot(momentFeature[:, 0], label="Red Mean")
ax[2].plot(momentFeature[:, 4], label="Green Mean")
ax[2].plot(momentFeature[:, 8], label="Blue Mean")
ax[2].set_xlabel("Iter")
ax[2].grid(True)
ax[2].legend(loc="upper right")

ax[3].plot(momentFeature[:, 1], label="Red Variance")
ax[3].plot(momentFeature[:, 5], label="Green Variance")
ax[3].plot(momentFeature[:, 9], label="Blue Variance")
ax[3].set_xlabel("Iter")
ax[3].grid(True)
ax[3].legend(loc="upper right")

ax[4].plot(momentFeature[:, 2], label="Red Skweness")
ax[4].plot(momentFeature[:, 6], label="Green Skweness")
ax[4].plot(momentFeature[:, 10], label="Blue Skweness")
ax[4].set_xlabel("Iter")
ax[4].grid(True)
ax[4].legend(loc="upper right")

ax[5].plot(momentFeature[:, 3], label="Red kurtosis")
ax[5].plot(momentFeature[:, 7], label="Green Kurtosis")
ax[5].plot(momentFeature[:, 11], label="Blue Kurtosis")
ax[5].set_xlabel("Iter")
ax[5].grid(True)
ax[5].legend(loc="upper right")

plt.show()
