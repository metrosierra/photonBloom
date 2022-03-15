#!/usr/bin/env python3
#%%
import numpy as np
import matplotlib.pyplot as plt

import os
from natsort import natsorted
import subroutines.sigbucket_subroutine as siggy
import subroutines.mathematics as mathy

data_dir = 'data/'
folders = natsorted(os.listdir(data_dir))
try: 
    folders.remove('archive')
except:
    pass
print(folders)

###because of natsorted (natural sort), collected_blah.npy should be
###in front of tags_blah.npy because alphabetical order

dossier = folders[1]
files = natsorted(os.listdir(data_dir + dossier))

tags = np.load(data_dir + dossier + '/' + files[0])
tags_channel_list = np.load(data_dir + dossier + '/' + files[1])


channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags, tags_channel_list)


index = 500
width = 1e3
signo = 50
peno = 19999*2

for i in range(5):

    peno = (i + 19999)*2

    time = (np.max(channel1[:index]) - np.min(channel1)) / 1e12
    print(time, 'time')
    output1 = siggy.signal_bin_combing(channel1[:index], bin_width = width, sig_bin_no = signo, period_no = peno)
    output2 = siggy.signal_bin_combing(channel2[:index], bin_width = width, sig_bin_no = signo, period_no = peno)

    plt.plot(output1, label = 'channel1')
    plt.legend()
    plt.xlabel('Combing index')
    plt.ylabel('Average signal bin count')
    # plt.xlim([97500, 102500])
    plt.savefig('output/channel1_{}_bin{}_sig{}_period{}_seconds{:.3g}.eps'.format(dossier, width, signo, peno, time), bbox = 'tight')
    plt.show(block = False)
    plt.close()

# plt.plot(output2, label = 'channel2')
# plt.legend()
# plt.xlabel('Combing index')
# plt.ylabel('Average signal bin count')
# # plt.xlim([97500, 102500])
# plt.savefig('output/channel2_{}_bin{}_sig{}_period{}_seconds{:.3g}.eps'.format(dossier, width, signo, peno, time), bbox = 'tight')
# plt.show(block = False)
# plt.close()
#%%