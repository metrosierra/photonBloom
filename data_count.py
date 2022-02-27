#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Feb 24 11:30:23 2022

@author: sabrinaperrenoud
"""


import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.stats import spearmanr
from collections import Counter
from scipy import optimize
from scipy.optimize import curve_fit
import matplotlib.cm as cm
from itertools import chain
import os
from natsort import natsorted

'''
Import data files
'''

#data_dir = 'data/'
#files = natsorted(os.listdir(data_dir))

#background_tags = np.load(data_dir + files[0])
#tags_channel_list = np.load(data_dir + files[1])


'''
Signal data
'''
data_tags = np.load('data/1uswidth_1khz_4vpp2voffset/collected_tags_25022022_18_05_18.npy')
Dtags_channel_list = np.load('data/1uswidth_1khz_4vpp2voffset/tags_channel_list_25022022_18_05_18.npy')

#data_tags = np.load('data/500ns_5khz_4vpp2voffset/collected_tags_25022022_18_06_08.npy')
#Dtags_channel_list = np.load('data/500ns_5khz_4vpp2voffset/tags_channel_list_25022022_18_06_08.npy')

#data_tags = np.load('data/50nswidth_50khz_4vpp2voffset/collected_tags_25022022_18_07_07.npy')
#Dtags_channel_list = np.load('data/50nswidth_50khz_4vpp2voffset/tags_channel_list_25022022_18_07_07.npy')

'''
Background data
'''

#data_tags = np.load('data/darkcount/collected_tags_25022022_18_04_28.npy')
#Dtags_channel_list = np.load('data/darkcount/tags_channel_list_25022022_18_04_28.npy')



print('File Size = ',len(data_tags))
'''
Classifying data according to detector channel
Note:
    channel1 = detector 2
    channel2 = detector 1
'''
channel1 = []
channel2 = []
channel3 = []
channel4 = []

tags_channel_list = Dtags_channel_list

for index, tag in enumerate(data_tags):

    if tags_channel_list[index] == 1.:
        channel1.append(tag)

    if tags_channel_list[index] == 2.:
        channel2.append(tag)

data = [channel1,channel2]

'''
Set data collection and signal parameters
'''

total_time = 30E12 #in picoseconds
time_interval = 3E12 #in picoseconds
time_crop = int(total_time / time_interval)

period = 1E9 #in picoseconds 
pulse_width = 1E6 #in picoseconds
bin_width = pulse_width * 2
bin_number = int(time_interval/bin_width)

print('Bin number = ',bin_number)

'''
Take data slice over time interval = time_interval
'''

crop_data=np.array([data[0][:int(len(data[0])/time_crop)] , data[1][:int(len(data[1])/time_crop)]])

'''
Extracting signal from raw data:
1- Splitting data over cycles of the signal period in 'data_array'
2- Stacking each array over the same period cycle in 'data_merge'
3- Taking histogram of resulting overlapped data
'''

data_lists = []
data_arrays = []

for index, channel in enumerate(crop_data):
    
    channel_no = str(index+1)
    
    data_array = [[]]
    
    start = data[index][0]
    zero_value = min(data[index]) 
    
    for x in channel:
        if x-start <= period+pulse_width:
            data_array[-1].append(x-start)
        else:
            data_array.append([x-start])
            start += period+pulse_width    
    data_lists.append(data_array)  #save each array of data

    data_merge = list(chain(*data_array))
    data_merge.sort()
    
    data_arrays.append(data_merge)  #save combined arrays of data
    
    '''
    Plotting histogram
    '''

    a,b = np.histogram(data_merge,bins=bin_number)
    
    detections=sum(a)
    print(detections,' detections in {time} seconds for channel {channel}'.format(time=int(time_interval*1E-12),channel=index+1))
    
    plt.plot(b[:-1],a,color=cm.autumn(index/2))
    
    plt.title('Histogram for Channel {channel} with bin={bins} over {time} second'.format(channel=index+1,bins=bin_number,time=time_interval*1E-12))
    plt.xlim(0,period*2)
    plt.grid()
    plt.savefig('output/histogram_1uswidth_1khz_4vpp2voffset_channel'+channel_no+'.eps')
    plt.show()
