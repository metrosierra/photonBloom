#!/usr/bin/env python3
import numpy as np
from numba import njit

import matplotlib.pyplot as plt

import subroutines.sigbucket_subroutine as siggy

@njit
def sig_chops_multiplex(datachops, chop_no, binwidth, sig_bin_no, sig_threshold, period_no, multiplex):

    output_bucket = [0]
    for i in range(chop_no):

        events = len(datachops[0])
        print(events, 'expected number of events1')
    
        print('Chop', i)
        comb_output = siggy.signal_bin_combing(datachops[i], bin_width = binwidth, sig_bin_no = sig_bin_no, period_no = period_no)

        shift = np.where(comb_output == np.max(comb_output))[0][0]
        print(shift,'shift!!!')
        exp_counts, is_signal = siggy.signal_bucketing(datachops[i], bin_width = binwidth, sig_bin_no = sig_bin_no, period_no = period_no, index_offset = shift, signal_threshold = sig_threshold, multiplex = multiplex)

        # hi1 = plt.hist(exp_counts)
        # plt.show()
        # print(hi1[0])


        # plt.plot(comb_output, label = 'channel1')
        # plt.axvline(x = shift, color = 'red')

        # plt.xlabel('Combing index')
        # plt.ylabel('Average signal bin count')
        # plt.show()


        output_bucket += [np.sum(run) for run in is_signal]

    return np.array(output_bucket[1:], dtype = np.int32)