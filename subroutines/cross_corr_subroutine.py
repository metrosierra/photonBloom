#!/usr/bin/env python3
import numpy as np
from numba import njit, prange

import subroutines.mathematics as mathy

###test
@njit(parallel = True)
def signal_bin_combing(data, bin_width, sig_bin_no = 1, period_no = 100):

    total_time = np.max(data) - np.min(data)
    # print(total_time/1e12, 'time')
    bin_no = int(np.floor(total_time / bin_width))

    period_cycles = int(np.floor(bin_no / period_no))
    print(period_cycles, 'cycles')
    counts, edges = mathy.numba_histogram(data, bin_no)

    output_data = []
    for i in range(period_no):

        sig_count = 0
        for q in prange(period_cycles-1):

            start_index = q*(period_no-1) + i
            sig_count += np.sum(counts[start_index : start_index + sig_bin_no])

        output_data.append(sig_count/period_cycles)

    return np.array(output_data, dtype = np.float64)

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
    


