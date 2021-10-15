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


data_stress = pd.read_csv('AlPressure.csv', header=0,usecols = ["o.sxx","o.syy","o.szz",
                                                           "o.sxy","o.syz","o.szx"])
data_stress = np.array(data_stress)
data = models.values_list('composition_id','output__volume_pa','energy_pa')
X1=[] # fully elements
y1=[] # volume
y2=[] # energy
for c,v,e in data:
    if v != None and e != None:
        X1.append(get_composition_descriptors(c).values())
        y1.append(v)
        y2.append(e)

# save data
X3=np.concatenate((data_stress,X1),axis=1)
y = [y1,y2]
np.savez('Al_multitasking118.npz', X=X3, y=y)