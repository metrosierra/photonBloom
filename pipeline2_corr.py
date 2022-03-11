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
data_dir = 'data/run2/'
folders = natsorted(os.listdir(data_dir))
try:
    folders.remove('archive')
except:
    pass
print(folders)
###because of natsorted (natural sort), collected_blah.npy should be
###in front of tags_blah.npy because alphabetical order

dossier = folders[0]
files = natsorted(os.listdir(data_dir + dossier))

tags = np.load(data_dir + dossier + '/' + files[0])
tags_channel_list = np.load(data_dir + dossier + '/' + files[1])


channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags, tags_channel_list)
data=[channel1,channel2]
channel1, channel2 = deli.data_crop(data, 30e12)

bins = 1000000
max_delay = 1100e6

counts, midpoints = cross.cross_corr(channel1, channel2, bins = bins, max_delay = max_delay)
midpoints *= 1e-6
plt.plot(midpoints, counts)
plt.xlabel('Delay ($\mu$s)')
plt.ylabel('Count')
plt.show()
plt.savefig('output/cross_corr_channel12_{}_bin{}_maxdelay{}.png'.format(dossier, bins, max_delay))

plt.plot(midpoints, counts)
plt.xlabel('Delay ($\mu$s)')
plt.ylabel('Count')
plt.xlim([-10, 10])
plt.show()

plt.savefig('output/cross_corr_channel12_{}_bin{}_maxdelay{}_zoomed.png'.format(dossier, bins, max_delay))

#%%

from scipy.signal import find_peaks
import subroutines.odr_fit as oddy

peak_no = 15
peaks, peaks_aux = find_peaks(counts, distance = 3e5)
print(len(peaks))
peaks_x = np.array([midpoints[index] for index in peaks])


partition = int(np.floor(len(midpoints)/peak_no/2.5))
print(partition, 'part')

fit_means = []
fit_means_err = []
for peak_index in peaks:

    print(peak_index)
    ydataslice = counts[peak_index - partition: peak_index + partition]
    xdataslice = midpoints[peak_index - partition: peak_index + partition]
    mean_guess = midpoints[peak_index]
    sigma_guess = np.std(ydataslice)/5e2
    print(sigma_guess, 'sigma')
    amp_guess = counts[peak_index] * (np.sqrt(2*np.pi)*0.02)
    print(amp_guess)

    fitoutput, fitstats, chi_red = oddy.fit(mathy.gaussian, xdataslice, ydataslice, initials = [mean_guess, 0.02, amp_guess])

    fit_means.append(fitoutput[0])
    fit_means_err.append(fitoutput[1])


    xfit  = np.linspace(xdataslice[0], xdataslice[-1], len(xdataslice))
    yfit = mathy.gaussian(fitoutput, xdataslice)
    plt.plot(xfit, yfit)
    plt.plot(xdataslice, ydataslice)
    plt.xlim([mean_guess - 20, mean_guess + 20])
    plt.show()


final_delay = np.average(np.diff(fit_means))
final_delay_err = np.sqrt(np.average(np.array(fit_means_err)**2)) 
print(final_delay)
print(final_delay_err)

