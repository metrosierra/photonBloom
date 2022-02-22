#!/usr/bin/env python3

import sys
import TimeTagger
import socket

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange



@njit(parallel = True)
def delay_cross_correlation(master, slave, max_delay = -1):

    length_master = len(master)
    length_slave = len(slave)
    delays = []



    origin = master[0]
    master = master - origin
    slave = slave - origin
    # print(master)
    # print(slave)

    for i in range(length_master):
        ref = master[i]
        output = slave - ref
        temp = output[output < max_delay]
        temp = temp[temp > -max_delay]
        delays.append(output)




    return delays
