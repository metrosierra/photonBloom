#!/usr/bin/env python3
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import os
from natsort import natsorted

import sys 
sys.path.append('../')
from subroutines import mathematics as mathy
from subroutines import twomultiplex_macroroutine as twosie
from subroutines import retrofit_macroroutine as rexy
from subroutines import cross_corr_subroutine as cross
from subroutines import prettyplot as plot
from subroutines import delay_tracking as deli


#%%
'''
We should treat this script as a jupyter notebook style, using vscode's cell+terminal functionality
to run the code. 

All data henceforth should be hard-coded into the script, with the methods/parameters of how they were
obtained being described in the comments explicitly and completely.

The plotting protocol should be called from prettyplots.py, with allowance for the aspect ratio of the plot, 
otherwise the formatting for all plots should be consistent.

We will stick to .eps format, and the output should be saved to output/keyplots, with appropriate relative 
file paths. 

If we want to use different data, we should state the different parameters in comments above the hardcoded array/list
(straightforward examples below). The point is to separate the data processing script (takes very long) and the plotting script (fast).
This makes the final report crunch much faster.

'''

#%%

'''

keyplots:

Sabrina:
photon8 data only (showing saturation) -> 60k, 100k, 140k 
photon4 data only (showing saturation) -> 20k, 70k, 150k
photon16 bidirectional histogram (using all 10s of data)-> 150k --> pipeline2_correlation_plot.py --> with function cross.cross_corr

Mingsong:
photon16 data + fit -> 50k, 100k, 150k, 200k
photon16 combing ladder -> 150k -> pipeline4_multiplex.py
photon16 jitter smear plot (varying size of data chop) -> 150k -> pipeline4_multiplex.py



'''

'''
- square plots generally look nice (9x9 plot vs fontsize 20 scaling)
- use inkscape or something to combine eps plots
- inward pointing ticks, for all four sides
- use different linestyle and colour for each line in plot
- use axis labels 'Counts' vs 'Clicks'
- significants figures/decimals reflect precision of quantity measured. else default to 3 sig fig (eg probability)
- errors are reported to ONE SIG FIG, value rounded to error precision
- legends go to largest empty space 
- legends: 'Click Distribution' and 'Poissonian Fit, mean'



'''



#%%


overall_qe = 0.589###AS OF 26 APRIL 2022 

#%%



###data
### fig, ax = prettyplot()
### plt.plot 
### plt.save 
### plt.show
### plt.close()

#%%

'''
Photon4 data only (showing saturation) -> 50k, 70k, 150k
'''
photon_no = 4
xaxis = np.arange(0, photon_no+1)
'''
Photon4 Data: 50k, 70k, 150k
'''
photon4_50k = [4, overall_qe, 0.006563, [508., 1082., 663., 215., 15.]]
photon4_70k = [4, overall_qe, 0.006385, [78., 536., 1179., 615., 77.]] 
photon4_150k = [4, overall_qe, 0.027358, [5., 87., 865., 1223., 309.]]

data_photon4 = [photon4_50k, photon4_70k, photon4_150k]
filename_photon4 = ['photon4_50k_histogram', 'photon4_70k_histogram', 'photon4_150k_histogram']

'''
Histogram Plot
'''

for i, data in enumerate(data_photon4):
    fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.1f', xaxis_dp = '%.0f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
    plt.bar(xaxis, data[3], width=1, color = 'lightgrey', edgecolor = 'black')
    plt.savefig('../output/{}.eps'.format(filename_photon4[i]))
    plt.show()
    plt.close()

'''
Line Plot
'''

for i, data in enumerate(data_photon4):
    fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.1f', xaxis_dp = '%.0f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
    plt.scatter(xaxis, data[3], color = 'black', marker='D')
    plt.plot(xaxis, data[3], color = 'black')
    plt.savefig('../output/{}.eps'.format(filename_photon4[i]+'_lineplot'))
    plt.show()
    plt.close()


#%%

'''
Photon8 data only (showing saturation) -> 60k, 100k, 140k
'''
photon_no = 8
xaxis = np.arange(0, photon_no+1)
'''
Photon8 Data: 60k, 100k, 140k
'''
photon8_60k = [8, overall_qe, 0.004672, [234., 661., 864., 493., 186., 37., 7., 1., 0.]]
photon8_100k = [8, overall_qe, 0.008244, [40., 232., 522., 775., 610., 239., 63., 5., 1.]]
photon8_140k = [8, overall_qe, 0.013272, [6., 44., 214., 527., 783., 595., 267., 51., 2.]]

data_photon8 = [photon8_60k, photon8_100k, photon8_140k]
filename_photon8 = ['photon8_60k_histogram', 'photon8_100k_histogram', 'photon8_140k_histogram']

'''
Histogram Plot
'''

for i, data in enumerate(data_photon8):
    fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.1f', xaxis_dp = '%.0f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
    plt.bar(xaxis, data[3], width=1, color = 'lightgrey', edgecolor = 'black')
    plt.savefig('../output/{}.eps'.format(filename_photon8[i]))
    plt.show()
    plt.close()

'''
Line Plot
'''

for i, data in enumerate(data_photon8):
    fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.1f', xaxis_dp = '%.0f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
    plt.scatter(xaxis, data[3], color = 'black', marker='D')
    plt.plot(xaxis, data[3], color = 'black')
    plt.savefig('../output/{}.eps'.format(filename_photon8[i]+'_lineplot'))
    plt.show()
    plt.close()

#%%
'''
photon16 bidirectional histogram (using all 10s of data)-> 150k --> pipeline2_correlation_plot.py --> with function cross.cross_corr
'''

photon16_150k = [16, overall_qe, 0.01725, [19., 102., 268., 503., 565., 444., 332., 172., 63., 15., 4., 1., 0., 0., 0., 0., 0.]]

photon_no = 16
xaxis = np.arange(0, photon_no+1)

'''
Loading and Saving Data
'''
# tags = np.load('../data/photon16/150k_countrate_50nsbench/50ns_50khz_4vpp2voffset/collected_tags_24032022_17_31_49.npy')
# tags_channel_list = np.load('../data/photon16/150k_countrate_50nsbench/50ns_50khz_4vpp2voffset/tags_channel_list_24032022_17_31_49.npy')
# channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags[:100000], tags_channel_list[:100000],commonfloor = True)
# data=[channel1,channel2]
# channel1, channel2 = deli.data_crop(data, 0.1e12)
# bins = 20000
# max_delay = 1100e6
# counts, midpoints = cross.cross_corr(channel1, channel2, bins = bins, max_delay = max_delay)
# midpoints *= 1e-6

# bidirectional_data = np.array([midpoints, counts])
# np.savetxt('../data/photon16/150k_data_bidirectional.txt', bidirectional_data)
'''
Plotting Bidirectional Histogram Photon16 150k countrate
'''

bidirectional_data = np.loadtxt('../data/photon16/150k_data_bidirectional.txt')

fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.1f', xaxis_dp = '%.2f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
plt.plot(bidirectional_data[0], bidirectional_data[1],color='red')
plt.xlabel('Delay ($\mu$s)')
plt.ylabel('Count')
plt.xlim([-10, 10])
plt.savefig('..output/photon16_150kcounts_bidirectional_histogram.eps')
plt.show()

#%%
