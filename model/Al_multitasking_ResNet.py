#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Train data using DNN/ResNet

from keras.models import Sequential,Model
from keras.layers import Dense, Dropout,BatchNormalization,Input,add # , LSTM, Activation
from keras.optimizers import Adam, Adadelta
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np
import time

# Read data and split data
time_start1=time.time()
data = np.load('Al_multitasking118.npz')
X = data['X']
y = data['y'].T
train_x, test_x, train_y, test_y = train_test_split(X, y, train_size=0.9)
input_layer = Input(shape=(train_x.shape[1], ))
dense1 = Dense(118, init='uniform', activation='relu')(input_layer)
dense1 = BatchNormalization()(dense1)
dense2 = Dense(128, activation='relu')(dense1)
dense3 = Dense(128, activation='relu')(dense2)
dense4 = Dense(118,  activation='relu')(dense3)
z1 = add([dense1, dense4])
dense5 = Dense(64, activation='relu')(z1)
dense6 = Dense(32, activation='relu')(dense5)
dense7 = Dense(32, activation='relu')(dense6)
dense7 = Dropout(0.25)(dense7)
out_layer = Dense(2,  activation='linear')(dense7)
model1 = Model(inputs=input_layer, outputs=out_layer)
adamoptimizer1 = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, decay=0.00001)
model1.compile(optimizer=adamoptimizer1, loss='logcosh')
history1= model1.fit(train_x, train_y, epochs=300, batch_size=256)
predict_y1 = model1.predict(test_x, batch_size=1)
time_end1=time.time()
# Summarize the results
volume_MAE = mean_absolute_error(test_y.T[0].T,predict_y1.T[0].T)
print(volume_MAE)
energy_MAE = mean_absolute_error(test_y.T[1].T,predict_y1.T[1].T)
print(energy_MAE)
volume_Rs = 1 - (mean_squared_error(test_y.T[0].T,predict_y1.T[0].T)/np.var(test_y.T[0].T))
print(volume_Rs)
energy_Rs = 1 - (mean_squared_error(test_y.T[1].T,predict_y1.T[1].T)/np.var(test_y.T[1].T))
print(energy_Rs)
print('totally cost time:',time_end1-time_start1)