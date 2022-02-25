#!/usr/bin/env python3

import sys
import TimeTagger
import socket

import numpy as np
import matplotlib.pyplot as plt
import time
from numba import njit, prange
from natsort import natsorted
import os

import mathematics as mathy


@njit
def signal_bin_combing(data, bin_width, sn_bin_no = (1, 99)):

    total_time = np.max(data) - np.min(data)
    bin_no = int(np.floor(total_time / bin_width))

    signal_no = sn_bin_no[0]
    noise_no = sn_bin_no[1]
    period_no = int(signal_no + noise_no)

    period_cycles = int(np.floor(bin_no / period_no))
    print(period_cycles, 'cycles')
    counts, edges = mathy.numba_histogram(data, bin_no)

    output_data = []
    for i in range(period_no*2):

        sig_count = 0
        for q in range(period_cycles-1):

            start_index = q*(period_no-1) + i
            sig_count += np.sum(counts[start_index : start_index + signal_no])

        output_data.append(sig_count / period_cycles)
    # print(output_data)
    return output_data



data_dir = 'data/'
files = natsorted(os.listdir(data_dir))

###because of natsorted (natural sort), collected_blah.npy should be
###in front of tags_blah.npy because alphabetical order

background_tags = np.load(data_dir + files[0])
tags_channel_list = np.load(data_dir + files[1])

channel1 = []
channel2 = []

for index, tag in enumerate(background_tags):

    if tags_channel_list[index] == 1.:
        channel1.append(tag)
    if tags_channel_list[index] == 2.:
        channel2.append(tag)

channel1 = np.array(channel1)
channel2 = np.array(channel2)

# print(channel1[:100], channel2[:100])

# #%%%%%%%%%%%%%%%%%%
#
# output1 = signal_bin_combing(channel1[:100000], bin_width = 0.5e5, sn_bin_no = (50, 450))
# output2 = signal_bin_combing(channel2[:100000], bin_width = 0.5e5, sn_bin_no = (50, 450))
#
# output1 = np.array(output1)
# output2 = np.array(output2)
# plt.plot(output1, label = 'channel1')
# plt.plot(output2, label = 'channel2')
# plt.legend()
# plt.xlabel('Combing index')
# plt.ylabel('Average signal bin count')
# plt.savefig('output/duochannel_fine_combing_test1.eps', bbox = 'tight')
# plt.show()

#%%%%%%%%%%%%%%%%%%
output1 = signal_bin_combing(channel1[:100000], bin_width = 0.5e5, sn_bin_no = (50, 450))
output2 = signal_bin_combing(channel1[100000:200000], bin_width = 0.5e5, sn_bin_no = (50, 450))

output1 = np.array(output1)
output2 = np.array(output2)
plt.xlabel('Combing index')
plt.ylabel('Average signal bin count')
plt.plot(output1, label = 'channel1_firsthalf')
plt.savefig('output/channel11_fine_combing_periodicity_test1.eps', bbox = 'tight')
plt.show()
plt.legend()
plt.xlabel('Combing index')
plt.ylabel('Average signal bin count')
plt.plot(output2, label = 'channel1_secondhalf')
plt.legend()
plt.savefig('output/channel12_fine_combing_periodicity_test1.eps', bbox = 'tight')


# plt.show()
