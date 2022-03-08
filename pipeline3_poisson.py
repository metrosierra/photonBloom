#!/usr/bin/env python3
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import os
from natsort import natsorted

import subroutines.cross_corr_subroutine as cross
import subroutines.mathematics as mathy
import subroutines.delay_tracking as deli
import subroutines.sigbucket_subroutine as siggy

data_dir = 'data/'
folders = natsorted(os.listdir(data_dir))
try:
    folders.remove('archive')
except:
    pass
print(folders)
#%%
###because of natsorted (natural sort), collected_blah.npy should be
###in front of tags_blah.npy because alphabetical order

dossier = folders[1]
files = natsorted(os.listdir(data_dir + dossier))

tags = np.load(data_dir + dossier + '/' + files[0])
tags_channel_list = np.load(data_dir + dossier + '/' + files[1])


channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags, tags_channel_list)
data = [channel1,channel2]
channel1, channel2 = deli.data_crop(0.05e12, data)

min_channel1 = min(channel1)
max_channel1 = max(channel1)

min_channel2 = min(channel2)
max_channel2 = max(channel2)


if min_channel1 < min_channel2:

    channel2 = np.insert(channel2, 0, min_channel1)

elif min_channel2 < min_channel1:

    channel1 = np.insert(channel1, 0, min_channel2)

if max_channel1 < max_channel2:

    channel1 = np.append(channel1, max_channel2)

elif max_channel2 < max_channel1:

    channel2 = np.append(channel2, max_channel1)

min_channel1 = min(channel1)
max_channel1 = max(channel1)

min_channel2 = min(channel2)
max_channel2 = max(channel2)
print(len(channel1), 'hiiii')
#%%
channel1 = np.array(channel1)
channel2 = np.array(channel2)

width = 1e3
signo = 80
peno = 20000*2


output1 = cross.signal_bin_combing(channel1, bin_width = width, sig_bin_no = signo, period_no = peno)

output2 = cross.signal_bin_combing(channel2, bin_width = width, sig_bin_no = signo, period_no = peno)
indexcut = 4000
output1 = output1[indexcut:]
output2 = output2[indexcut:]

#%%
print(np.max(output1))
print(np.max(output2))
shift1 = np.where(output1 == np.max(output1))[0][0] + indexcut 
shift2 = np.where(output2 == np.max(output2))[0][0] + indexcut 

experiments1 = siggy.signal_bucket(channel1, bin_width = width, sig_bin_no = signo*100, period_no = peno/2, index_offset = 0)

experiments2 = siggy.signal_bucket(channel2, bin_width = width, sig_bin_no = signo*100, period_no = peno/2, index_offset = 0)

aggregate = experiments1 + experiments2 
plt.hist(aggregate, 2)


# plt.plot(output1, label = 'channel1')
# plt.legend()
# plt.xlabel('Combing index')
# plt.ylabel('Average signal bin count')
# # plt.xlim([19000, 22000])
# plt.show()

# plt.plot(output2, label = 'channel2')
# plt.legend()
# plt.xlabel('Combing index')
# plt.ylabel('Average signal bin count')
# # plt.xlim([19000, 22000])
# plt.show()





#%%