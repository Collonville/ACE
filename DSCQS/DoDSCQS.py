"""
研究室のMatrixに画像を投影するpythonプログラム
python drawMatrix.py VAL.png
とやると，VAL.pngが投影される．
解像度が50x50出ない画像は自動的にリサイズされる．

モジュールとしてdrawMatrix関数を使う場合は，
解像度が50x50になっている事が前提なので注意
"""
import math
import socket
import sys
from ctypes import *
from time import sleep

import numpy as np

import cv2

inputImagePath = ["pict/ara-3695678__340.jpg",
                  "pict/s35_8.jpg",
                  "pict/s496_1.jpg",
                  "pict/strawberry.jpg"]

enhancedImagePath = ["pict/ara-3695678__340_31_k08_Blend_Final.jpg",
                     "pict/s35_8_39_k08_Blend_Final.jpg",
                     "pict/s496_1_40_k08_Blend_Final.jpg",
                     "pict/strawberry_38_k08_Blend_Final.jpg"]

class Artnet:
    # Artnetパケットを送信するためのクラス
    class ArtNetDMXOut(LittleEndianStructure):
        PORT = 0x1936
        _fields_ = [("id", c_char * 8),
                    ("opcode", c_ushort),
                    ("protverh", c_ubyte),
                    ("protver", c_ubyte),
                    ("sequence", c_ubyte),
                    ("physical", c_ubyte),         
                    ("universe", c_ushort),
                    ("lengthhi", c_ubyte),
                    ("length", c_ubyte),
                    ("payload", c_ubyte * 512)]
        
        def __init__(self):
            self.id = b"Art-Net"
            self.opcode = 0x5000
            self.protver = 14
            self.universe = 0
            self.lengthhi = 2

    def __init__(self):
        self.artnet = Artnet.ArtNetDMXOut()
        self.S = socket.socket(socket.AF_INET,socket.SOCK_DGRAM)    
        for i in range(512):
            self.artnet.payload[i] = 0

    def send(self,data,IP,port):
        # sendDMX(送るデータ,IPアドレス,ポート番号)
        self.artnet.universe = port
        for i in range(512):
            if(i < len(data)):
                self.artnet.payload[i] = int(data[i])
            else:
                break
        self.S.sendto(self.artnet,(IP,Artnet.ArtNetDMXOut.PORT))

    def off(self,IP):
        dmx = [0] * 512
        for i in range(6):
            self.artnet.universe = i
            self.S.sendto(self.artnet,(IP,Artnet.ArtNetDMXOut.PORT))

def draw(img, mode=0):
    """
    img:画像データ (cv.imread)
     50x50にリサイズされている事を前提としている

    mode: 0~1
     0:RGB値をそのまま出力
     1:RGB値をsRGBとして，Labに一旦変換し，LED制御用RGB値に変換して出力
    """

    if mode == 0:
        print('draw RAW RGB')
        use_img = img
    else:
        print('draw RAW RGB')
        use_img = img
    
    if img.shape[0] == 50 and img.shape[1] == 50:
        artnet = Artnet()
        dmx = np.zeros([3,6,512],int)
        IP = ['133.15.42.102','133.15.42.103','133.15.42.104']
        for i in range(50*50):
            y = i % 50
            x = int(i / 50)
            decoder = int(x/18)
            port = int((x%18)/3)
            line = (x%18)%3
        
            dmx[decoder,port,(line * 50 + y) * 3 + 0] = use_img[49-y,x,0]
            dmx[decoder,port,(line * 50 + y) * 3 + 1] = use_img[49-y,x,1]
            dmx[decoder,port,(line * 50 + y) * 3 + 2] = use_img[49-y,x,2]

        for decoder in range(3):
            for port in range(6):
                artnet.send(dmx[decoder][port],IP[decoder],port)

    else:
        print('No image or image size is not 50x50')
                
def off():
    """
    消灯する関数
    """
    artnet = Artnet()
    off = [0] * 512
    IP = ['133.15.42.102','133.15.42.103','133.15.42.104']
    for decoder in range(3):
        for port in range(6):
            artnet.send(off,IP[decoder],port)
       
def main():
    colormode = 0

    for round in range(4):
        print("------" + str(round) + " Round------")
        draw(cv2.cvtColor(cv2.imread(inputImagePath[round]), cv2.COLOR_BGR2RGB), colormode)
        sleep(10)
        print("Off lights")
        off()
        sleep(3)
        draw(cv2.cvtColor(cv2.imread(enhancedImagePath[round]), cv2.COLOR_BGR2RGB), colormode)
        sleep(10)
        print("Off lights")
        off()
        sleep(3)

        print("Start to grading")
        draw(cv2.cvtColor(cv2.imread(inputImagePath[round]), cv2.COLOR_BGR2RGB), colormode)
        sleep(10)
        print("Off lights")
        off()
        sleep(3)
        draw(cv2.cvtColor(cv2.imread(enhancedImagePath[round]), cv2.COLOR_BGR2RGB), colormode)
        sleep(10)
        print("Off lights")
        off()
        sleep(10)


if __name__ == '__main__':
    main()    
    print("End")
