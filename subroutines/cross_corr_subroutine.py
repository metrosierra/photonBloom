#!/usr/bin/env python3
import numpy as np
from numba import njit, prange

import subroutines.mathematics as mathy

###test


@njit
def cross_corr(master, slave, bins, max_delay):

    output = np.array([1.])
    for start in master:

        delays = -slave + start 
        delays = delays[delays < max_delay]
        delays = delays[delays > -max_delay]
        output = np.concatenate((output, delays))

    output = output[1:].ravel()
    counts, edges = mathy.numba_histogram(output, bins)
    midpoints = edges[:-1] + (edges[1] - edges[0])/2

    return counts, midpoints
    


