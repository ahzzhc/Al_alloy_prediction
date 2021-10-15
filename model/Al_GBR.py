#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Train data using GBR

from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import matplotlib.pyplot as plt
import numpy as np
import time
from sklearn.metrics import mean_absolute_error

# load data
data = np.load('Al_volume118_split.npz',allow_pickle=True)
train_x=data["train_x"]
train_y=data["train_y"]
test_x=data["test_x"]
test_y=data["test_y"]
parameters = {
        'n_estimators': [10, 100, 500],
        'max_depth': [2,3,4],
        'min_samples_split': [2,3,4] ,
        'learning_rate': [0.001, 0.01, 0.1]}
gbr = GradientBoostingRegressor()
clf = GridSearchCV(gbr, parameters)  # Grid search optimal model

# test model
time_start1=time.time()
clf.fit(train_x, train_y)
predict_y = clf.predict(test_x)
MAE=mean_absolute_error(test_y,predict_y)
time_end1=time.time()
print('totally cost time:',time_end1-time_start1)
print(clf.score(test_x, test_y))
print(MAE)

# Draw the result diagram
plt.figure(figsize=(6,6))
plt.subplots_adjust(left=0.16, bottom=0.16, right=0.95, top=0.90)
plt.rc('font', family='Arial narrow')
plt.rcParams['savefig.dpi'] = 300 #图片像素
plt.rcParams['figure.dpi'] = 300 #分辨率
plt.title('GBR Model', fontsize=28, pad=12)
plt.tick_params(labelsize=26)
plt.ylabel('ML Prediction', fontname='Arial Narrow', size=28)
plt.xlabel('DFT Calculation', fontname='Arial Narrow', size=28)
plt.scatter(test_y,predict_y,c='orange',marker="*",edgecolors='dimgrey', alpha=1.0)
plt.plot(test_y,test_y)
plt.grid(False)
plt.show()
