#!/usr/bin/env python3
##%%%%%%%%%%%%%

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.stats import spearmanr

import os
from natsort import natsorted
import cross_corr_subroutine as cross


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

channel1 = channel1[:30000]
channel2 = channel2[:30000]

# print(channel1[:100], channel2[:100])

#%%%%%%%%%%%%%%%%%%
# output = cross.delay_cross_correlation(np.array(channel1), np.array(channel2), max_delay = 100000)

output = cross.delay_cross_correlation(np.array(channel1), np.array(channel2), max_delay = 1e15)
output = np.array(output)
# print(output)
# print(len(output))
# output = np.concatenate(output).ravel()
# print(output)
plt.hist(output.flatten(), bins = 100)
plt.show()
