#!/usr/bin/env python3

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
import json
import big_tagger

testag = TagClient()

for i in range(100):
    
    binwidth_ns=2.5
    n_values=200
    runtime=int(500e12)
    
    data=testag.get_correlation(channels=[3,4],binwidth=binwidth_ns,n=n_values,startfor=runtime)
    mydict={"bindwidth_ns":binwidth_ns, "n_values":n_values, "runtime_ps":runtime, "data":data.tolist()}
    output_object=json.dumps(mydict,indent=1)
    print(output_object)
    timestr = time.strftime("%Y%m%d-%H%M%S")
    #np.savetxt("C:/QMLab Experimental Data/photonBloom/"+timestr+".txt",output_object)
    jsonFile = open("F:/QM Lab Nanowire Data/CorrelationMeasurements/PhotonBloomCorrelation_"+timestr+".dat", "w")
    jsonFile.write(output_object)
    jsonFile.close()
   