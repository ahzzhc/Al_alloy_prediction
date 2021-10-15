#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Build training data based on composition and pressure

from qmpy import *
import numpy as np
import pandas as pd

# Select Al alloy material
models = Calculation.objects.filter(path__contains='icsd')
models = models.filter(converged=True, label__in=['static', 'standard'])
models = models.filter(composition__element_set="Al")

#create Al alloy dataset
data_stress = pd.read_csv('AlPressure.csv', header=0,usecols = ["o.sxx","o.syy","o.szz",
                                                           "o.sxy","o.syz","o.szx"])
data_stress = np.array(data_stress)
data = models.values_list('composition_id','energy_pa')
y1=[] # energy data
X1=[] # composition data
for c,e in data:
    if e != None:
        X1.append(get_composition_descriptors(c).values())
        y1.append(e)

# save data
X3=np.concatenate((X1,data_stress),axis=1)
np.savez('Al_energy118.npz', X=X3, y=y1)
