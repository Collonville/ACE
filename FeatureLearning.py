import codecs
import copy
import glob
import itertools
import os
import sys

import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numba
import numpy as np
import sklearn
import win_unicode_console
from mpl_toolkits.mplot3d import Axes3D
from scipy import linalg, signal
from scipy.optimize import *
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
import ImageFeature

win_unicode_console.enable()

#表示のフォーマットを定義
np.set_printoptions(precision=10, suppress=True, threshold=np.inf, linewidth=100)

"""
ファイル名と特徴量の辞書(ImageFeaturesWithPath)からファイル名でソートされた順の特徴量を得る
"""
def getSortedImageFeatures():
    imageFeatures = np.load("ImageFeature/ImageFeaturesWithPath_5000image_40dim_20181219_130703.npy")

    features = []
    
    for fileName in [os.path.basename(r) for r in glob.glob('img/All/*.jpg')]:
        #拡張子とファイル名を分割
        filename = fileName.split('.')

        for index in range(100):
            #インデックス値を挿入
            pathKey = filename[:][0] + "_" + str(index) + "." + filename[:][1]

            #ファイル名から特徴量の取得
            imageFeature = imageFeatures.item().get('outimg/continuity_hue/All\\' + pathKey)[0]

            features.append(imageFeature)

    return np.array(features)

def getTrainingdata():
    trainingFiles = glob.glob("TrainingData/*.csv")

    #画像特徴量取得に関するインスタンス
    #imageFeature = ImageFeature.ImageFeature()
    #imageFeature.getFeaturesFromPath("outimg/continuity_hue/All/*.jpg", [])

    #画像特徴量の取得([5000, 特徴量次元数])
    imageFeatues = getSortedImageFeatures()
    print("Sorted Image Features size : %s" % (str(imageFeatues.shape)))

    trainingLabel = np.empty(0, dtype=int)
    trainingFeature = np.empty((0, imageFeatues.shape[1]))
    
    #それぞれの教師データを取得
    for fileName in trainingFiles:
        matrix = np.loadtxt(fileName, delimiter=",")

        #特徴量の追加
        trainingFeature = np.r_[trainingFeature, imageFeatues]

        #ラベルの追加
        for imageIdx in range(50):
            selected = matrix[imageIdx, :]
            trainingLabel = np.append(trainingLabel, selected, axis=0)
    
    return trainingFeature, trainingLabel

feature, label = getTrainingdata()

#nanがある場合は0で埋める(要修正)
feature[np.isnan(feature)] = 0

#------------------------------------------------------------------------
#訓練とテストデータを割合で分割
'''
X_train, X_test, y_train, y_test = train_test_split(feature, label, test_size=0.1)

print(X_train.shape)
sys.exit()'''
trainNum = 15000
testNum = label.shape[0] - trainNum

print("Dataset num:%d, Train num:%d, Test num:%d" % (label.shape[0], trainNum, testNum))

#ランダムでインデックスを作成
randomIdx = np.random.choice(label.shape[0], label.shape[0], replace=False)
trainIdx = randomIdx[0:trainNum]
testIdx = randomIdx[trainNum:label.shape[0]]

#訓練とテストデータに分ける
trainFeature = feature[trainIdx]
trainLabel   = label[trainIdx]
testFeature  = feature[testIdx]
testLabel    = label[testIdx]

#------------------------------------------------------------------------
#特徴量の正規化
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(trainFeature)
trainFeature = scaler.transform(trainFeature)

#保存
sklearn.externals.joblib.dump(scaler, "LogisticRegresion/FeatureScaler.pkl", compress=True)

#------------------------------------------------------------------------
#学習の実行
lr = LogisticRegression(solver='lbfgs', tol=1e-6, max_iter=1000, C=0.01, random_state=123).fit(trainFeature, trainLabel)

#クロスバリデーションによる最適パラメータ決定
'''
#https://data-science.gr.jp/implementation/iml_sklearn_logistic_regression.html
diparameter={"C":[10**i for i in range(-2,4)],"random_state":[123],}
licv=sklearn.model_selection.GridSearchCV(LogisticRegression(),param_grid=diparameter,cv=5)
licv.fit(trainFeature, trainLabel)
predictor=licv.best_estimator_

print(licv.best_score_)  # 最も良かったスコア
print(licv.best_params_)  # 上記を記録したパラメータの組み合わせ
print(predictor)
'''

#------------------------------------------------------------------------
#学習の結果の保存
np.save("LogisticRegresion/intercept", lr.intercept_)
np.save("LogisticRegresion/coef", lr.coef_[0])

#テストデータの正規化
testFeature = scaler.transform(testFeature)

print("Training Score: %f" % (lr.score(trainFeature, trainLabel)))
print("Test Score: %f" % (lr.score(testFeature, testLabel)))
print("Intercept: %f" % (lr.intercept_))
print("Coef: %s" % (np.array2string(lr.coef_[0])))
