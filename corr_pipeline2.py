#!/usr/bin/env python3
#%%
import numpy as np
import matplotlib.pyplot as plt

import os
from natsort import natsorted
import cross_corr_subroutine as cross
import mathematics as mathy

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

bins = 100000
max_delay = 1e8

counts, midpoints = cross.cross_corr(channel1, channel2, bins = bins, max_delay = max_delay)
midpoints *= 1e-6
plt.plot(midpoints, counts)
plt.xlabel('Delay ($\mu$s)')
plt.ylabel('Count')

plt.savefig('output/cross_corr_channel12_{}_bin{}_maxdelay{}.png'.format(dossier, bins, max_delay), bbox = 'tight')
plt.xlim([-0.2, 0.2])
plt.savefig('output/cross_corr_channel12_{}_bin{}_maxdelay{}_zoomed.png'.format(dossier, bins, max_delay), bbox = 'tight')
