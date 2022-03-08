#!/usr/bin/env python3
import numpy as np
from numba import njit, prange

import subroutines.mathematics as mathy

###test
@njit
def signal_bucket(data, bin_width, sig_bin_no = 1, period_no = 100, index_offset = 0):

    total_time = np.max(data) - np.min(data)
    # print(total_time/1e12, 'time')
    bin_no = int(np.floor(total_time / bin_width))

    period_cycles = int(np.floor(bin_no / period_no))
    print(period_cycles, 'cycles')
    counts, edges = mathy.numba_histogram(data, bin_no)

    output_data = []

    for q in range(period_cycles-1):
        sig_count = 0
        start_index = q*(period_no-1) + index_offset
        sig_count += np.sum(counts[start_index : start_index + sig_bin_no])

        output_data.append(sig_count)

    return np.array(output_data, dtype = np.int64)
