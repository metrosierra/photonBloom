# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 17:03:24 2022

@author: Quantum Photonics
"""
import sys

sys.path.append('../')

from client_object import *
import numpy as np
import time


testag = TagClient()

testag.set_trigger(2,0.08)
testag.set_trigger(3,0.08)
testag.set_trigger(4,0.08)

for i in range(100):

    data=testag.triggered_correlation(binwidth_ns=100,n_values=100,runtime=int(500e12))
    
    timestr = time.strftime("%Y%m%d-%H%M%S")
    np.savetxt("C:/QMLab Experimental Data/photonBloom/output/"+timestr+".txt",data)