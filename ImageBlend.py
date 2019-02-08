import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import win_unicode_console
from PIL import Image

win_unicode_console.enable()

def getImageRGBFromPath(filePath):
    inputImg = cv2.imread(filePath, cv2.IMREAD_COLOR)

    #正規化と整形
    inputImg = cv2.cvtColor(inputImg, cv2.COLOR_BGR2RGB) / 255.
    rgb = np.reshape(inputImg, (inputImg.shape[0] * inputImg.shape[1], 3))

    return rgb, inputImg.shape[0], inputImg.shape[1]

#------------------------------------------------------------------------
enhanceImgpath = "outimg/ACE2/ACEMethod1/"
signalImgPath = "outimg/ACE2/ACEMethod1/Signal/"
blendImgOutputPath = "outimg/ACE2/ACEMethod1/Blend/"
imgFilename = sys.argv[1]
iter = sys.argv[2]
k = float(sys.argv[3])

enhanceRGB, enhanceImgH, enhanceImgW = getImageRGBFromPath(enhanceImgpath + imgFilename + "_" + str(iter) + ".jpg")
signalRGB,  signalImgH,  signalImgW  = getImageRGBFromPath(signalImgPath + imgFilename + "_" + str(iter) + "_Signal.jpg")

if (enhanceImgH != signalImgH) | (enhanceImgW != signalImgW):
    print("画像の縦横画素数が違います")
    sys.exit()

#ブレンド画像の生成
blendRGB = (1 - k) * enhanceRGB + k * signalRGB
blendRGB = np.clip(blendRGB, 0, 1)

#(H, W, 3)に変形
enhanceRGB = enhanceRGB.reshape((enhanceImgH, enhanceImgH, 3))
signalRGB  = signalRGB.reshape((signalImgH, signalImgH, 3))
blendRGB   = blendRGB.reshape((enhanceImgH, enhanceImgW, 3))

fig = plt.figure()
ax1 = fig.add_subplot(131)
ax1.imshow(enhanceRGB)
ax1.set_title("Enhanced Image, Iter=" + str(iter))

ax2 = fig.add_subplot(132)
ax2.imshow(signalRGB)
ax2.set_title("Signal Image, Iter=" + str(iter))

ax3 = fig.add_subplot(133)
ax3.imshow(blendRGB)
ax3.set_title("Blend Image, k=" + str(k))

im = Image.fromarray(np.uint8(blendRGB * 255))
im.save(blendImgOutputPath + imgFilename + "_" + str(iter) + "_k" + str(k).replace('.', '') + "_Blend.jpg", quality=100)

plt.show()
