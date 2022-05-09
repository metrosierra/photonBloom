#!/usr/bin/env python3
import numpy as np
from numba import njit

import matplotlib.pyplot as plt

import subroutines.prettyplot as pretty
import subroutines.sigbucket_subroutine as siggy

# @njit
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

        fig, ax = pretty.prettyplot(figsize = (9, 9), yaxis_dp = '%.0f', xaxis_dp = '%.0f', ylabel = 'Collected Signal Count', xlabel = 'Scan Offset (ns)', title = '')

        plt.plot(comb_output)
        # plt.axvline(x = shift, color = 'red')
        plt.savefig('../output/final_photon16_50k_comb_pyramid.eps', bbox_inches="tight")
        plt.savefig('../output/final_photon16_50k_comb_pyramid.png', dpi = 200, bbox_inches="tight")       
        plt.show()


        output_bucket += [np.sum(run) for run in is_signal]

    return np.array(output_bucket[1:], dtype = np.int32)