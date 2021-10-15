#!/usr/bin/python
# -*- coding: UTF-8 -*-

# Read pressure and elements

from qmpy import *
import csv

# Select Al alloy materials
models = Calculation.objects.filter(path__contains='icsd')
models = models.filter(converged=True, label__in=['static', 'standard'])
models = models.filter(composition__element_set="Al")

# Export the element and its pressure
f = open('AlPressure.csv','wb')
csv_writer = csv.writer(f)
csv_writer.writerow(["id","composition_id","o.sxx",
                     "o.syy","o.szz","o.sxy","o.syz","o.szx"])
for m in models:
    csv_writer.writerow([m.id,m.composition_id,m.output.stresses[0],
                         m.output.stresses[1], m.output.stresses[2],
                         m.output.stresses[3], m.output.stresses[4],
                         m.output.stresses[5],
                         ])
f.close()
exit()