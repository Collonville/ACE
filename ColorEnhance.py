import sys
from argparse import ArgumentParser

import colour
import matplotlib.pyplot as plt
import numpy as np
import win_unicode_console

import ACE

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=10, suppress=True, threshold=np.inf, linewidth=100)

def getOption():
    argparser = ArgumentParser()

    argparser.add_argument('-imgName', '--imgName', type=str,
                           default="",
                           help='Input image file name')
    argparser.add_argument('-imagePath', '--imagePath', type=str,
                           default="",
                           help='Input image file path')
    argparser.add_argument('-eImgPath', '--enhanceImagePath', type=str,
                           default="outimg/ACE2/ACEMethod1/",
                           help='Output path of enhanced image')
    argparser.add_argument('-sImgPath', '--signalImagePath', type=str,
                           default="outimg/ACE2/ACEMethod1/Signal/",
                           help='Output path of signal image')
    argparser.add_argument('-method', '--method', type=str,
                           default="ACE1",
                           help='Method type of ACE')

    return argparser.parse_args()

if __name__ == '__main__':
    args = getOption()

    fileName = str(args.imgName)

    energySet, hueLossSet = ACE.doEnhance(str(args.method), str(args.imagePath), fileName, str(args.enhanceImagePath), str(args.signalImagePath))

    #------------------------------------------------------------------------
    fig = plt.figure()
    fig.suptitle(fileName + ", " + str(args.method), fontsize=12, fontweight='bold')

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
