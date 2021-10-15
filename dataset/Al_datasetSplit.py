#!/usr/bin/python
# -*- coding: UTF-8 -*-

from sklearn.model_selection import train_test_split
import numpy as np

# formationenergy118 split
data = np.load('Al_formationenergy118.npz',allow_pickle=True)
x=data["X"]
y=data["y"]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.1)
np.savez('Al_formationenergy118_split.npz', train_x=train_x, train_y=train_y,test_x=test_x,test_y=test_y)

# volume118 split
data = np.load('Al_volume118.npz',allow_pickle=True)
x=data["X"]
y=data["y"]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.1)
np.savez('Al_volume118_split.npz', train_x=train_x, train_y=train_y,test_x=test_x,test_y=test_y)

# energy118 split
data = np.load('Al_energy118.npz',allow_pickle=True)
x=data["X"]
y=data["y"]
train_x, test_x, train_y, test_y = train_test_split(x, y, test_size = 0.1)
np.savez('Al_energy118_split.npz', train_x=train_x, train_y=train_y,test_x=test_x,test_y=test_y)

