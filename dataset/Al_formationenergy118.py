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
data_id = pd.read_csv('AlPressure.csv', header=0,usecols = ["id"])
data_id = np.array(data_id)
data = models.values_list('id','composition_id',"formationenergy")
y1=[] # formationenergy data
X1=[] # stress data
X2=[] # composition data
for id,c,f in data:
    for i in range(len(data_id)):
        if id == data_id[i] and f != None:
            y1.append(FormationEnergy.objects.get(id = f).delta_e)
            X2.append(get_composition_descriptors(c).values())
            X1.append(data_stress[i])

# save data
X3=np.concatenate((X1,X2),axis=1)
np.savez('Al_formationenergy118.npz', X=X3, y=y1)