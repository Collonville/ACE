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
argv = sys.argv
feature = np.load(argv[1] + "_Features.npy")
momentFeature = np.load(argv[1] + "_MomentFeatures.npy")

fig, axes = plt.subplots(nrows=2, ncols=3)
ax = axes.ravel()

ax[0].plot(feature[:, 0])
ax[0].set_title("Average Gradient")
ax[0].grid(True)
ax[0].legend()
ax[0].set_xlabel("Iter")

#エントロピー
ax[1].plot(feature[:, 1], label="Hue")
ax[1].plot(feature[:, 2], label="Saturation")
ax[1].plot(feature[:, 3], label="Luminance")
ax[1].set_title("HSL Entropy")
ax[1].set_xlabel("Iter")
ax[1].grid(True)
ax[1].legend(loc="upper right")

#平均
ax[2].plot(momentFeature[:, 0], label="Red")
ax[2].plot(momentFeature[:, 4], label="Green")
ax[2].plot(momentFeature[:, 8], label="Blue")
ax[2].set_title(u"RGB Mean")
ax[2].set_xlabel("Iter")
ax[2].grid(True)
ax[2].legend(loc="upper right")

#分散
ax[3].plot(momentFeature[:, 1], label="Red")
ax[3].plot(momentFeature[:, 5], label="Green")
ax[3].plot(momentFeature[:, 9], label="Blue")
ax[3].set_title(u"RGB Variance")
ax[3].set_xlabel("Iter")
ax[3].grid(True)
ax[3].legend(loc="upper right")

#歪度
ax[4].plot(momentFeature[:, 2], label="Red")
ax[4].plot(momentFeature[:, 6], label="Green")
ax[4].plot(momentFeature[:, 10], label="Blue")
ax[4].set_title(u"RGB Skweness")
ax[4].set_xlabel("Iter")
ax[4].grid(True)
ax[4].legend(loc="upper right")

#尖度
ax[5].plot(momentFeature[:, 3], label="Red")
ax[5].plot(momentFeature[:, 7], label="Green")
ax[5].plot(momentFeature[:, 11], label="Blue")
ax[5].set_title("RGB kurtosis")
ax[5].set_xlabel("Iter")
ax[5].grid(True)
ax[5].legend(loc="upper right")

plt.tight_layout()
plt.show()
