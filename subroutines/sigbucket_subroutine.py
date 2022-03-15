#!/usr/bin/env python3
import numpy as np
from numba import njit, prange

import subroutines.mathematics as mathy


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

        output_data.append(sig_count)

    return np.array(output_data, dtype = np.float64)



@njit
def signal_bucketing(data, bin_width, sig_bin_no = 1, period_no = 100, index_offset = 0, signal_threshold = 1, multiplex = 1, verbose = False):


    multiplex = int(multiplex)
    if multiplex > 1 and verbose:
        print('Higher order multiplexing setting turned ON, please ensure signal bucket (sig_bin_no) is fat enough')

    total_time = np.max(data) - np.min(data)
    # print(total_time/1e12, 'time')
    bin_no = int(np.floor(total_time / bin_width))

    period_cycles = int(np.floor(bin_no / period_no))
    print(period_cycles, 'periods included! Each period should have a signal bucket comprising', multiplex, 'pulse(s) at maximum')
    counts, edges = mathy.numba_histogram(data, bin_no)

    output_counts = np.zeros((period_cycles-1, multiplex), dtype = np.int64)
    is_signal = np.zeros((period_cycles-1, multiplex), dtype = np.int64)
    

    for q in range(period_cycles-1):
        start_index = q*(period_no-1) + index_offset

        subdomain = counts[start_index : start_index + sig_bin_no]
        multiplex_interval = round(len(subdomain) / multiplex)
        for i in range(multiplex):

            sig_count = np.sum(subdomain[i*multiplex_interval : (i+1)*multiplex_interval])
            output_counts[q][i] = sig_count
            if sig_count >= signal_threshold: 
                is_signal[q][i] = 1
            
            else: is_signal[q][i] = 0


    return output_counts, is_signal



