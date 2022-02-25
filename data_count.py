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

import os
from natsort import natsorted


'''
Data Stats

Channel 1 | Channel 2
Number of events = 191715 | 391931
Mean Count = 0.0064 | 0.013
Correlation = -0.000002 = NO correlation
Distribution mean = 20.6 | 40.5
Standard Deviation = 3.16 | 3.67
Distribtuion Peak = 907 | 640
'''
'''
Bakcground Data Stats

Channel 1 | Channel 2
Number of events = 191543 | 391110
Mean Count = 0.0064 | 0.013
Correlation = 0.007 = NO correlation
Distribution mean = 22.1 | 41.5
Standard Deviation = 3.03 | 3.72
Distribtuion Peak = 949 | 677
'''


'''

### the /data folder
#data_dir = 'data/'
#files = natsorted(os.listdir(data_dir))

###because of natsorted (natural sort), collected_blah.npy should be
###in front of tags_blah.npy because alphabetical order

#background_tags = np.load(data_dir + files[0])
#tags_channel_list = np.load(data_dir + files[1])
'''
background_tags = np.load('data/background_count/collected_tags_15022022_16_55_30.npy')
Btags_channel_list = np.load('data/background_count/tags_channel_list_15022022_16_55_30.npy')


data_tags = np.load('data/10microsecond_pulse/collected_tags_15022022_16_58_24.npy')
Dtags_channel_list = np.load('data/10microsecond_pulse/tags_channel_list_15022022_16_58_24.npy')

# print(background_tags)
#
# print(tags_channel_list)

print(len(data_tags))
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

c1_background = np.loadtxt('background_data_channel1_30s.txt')
c2_background = np.loadtxt('background_data_channel2_30s.txt')
background = [c1_background,c2_background]


tags_channel_list = Dtags_channel_list

for index, tag in enumerate(data_tags):

    if tags_channel_list[index] == 1.:
        channel1.append(tag)

    if tags_channel_list[index] == 2.:
        channel2.append(tag)

data = [channel1,channel2]
bin_value=3E7

histogram_data=[]
# bin_range=[3E3, 3E4, 3E5, 3E6, 3E7, 3E8]

for index, channel in enumerate(data):
    time = max(channel) - min(channel)
    tags_no = len(channel)
    print('Number of events = ',tags_no)

    counts, edges = np.histogram(channel,bins = int(bin_value))
    print('Mean count = ',np.mean(counts))
    histogram_data.append(counts)
    plt.plot(edges[:-1],counts)
    plt.title('Histogram of raw data with bin={bins} for channel {channel}'.format(bins=bin_value,channel=index+1))
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
'''
Number of events = 191715 | 391931
Mean Count = 0.0064 | 0.013
'''
    
#%%

'''
Taking the moving average of data before histogram binning
'''
averaged_data=[]
length=[]
for index, channel in enumerate(data):
    data_length = len(channel)
    interval = 1000
    i = 0
    average_data = []
    
    while i < data_length - interval + 1:
        data_interval = channel[i : i + interval]
        interval_average = sum(data_interval) / interval
        average_data.append(interval_average)
        i += interval
    averaged_data.append(average_data)
    
#%%  
from collections import Counter
averaged_histogram_data=[]
frequency=[]
for c in averaged_data:
    counts, edges = np.histogram(c,bins = int(bin_value))
    averaged_histogram_data.append(counts)
    
    measurements = len(c)
    counting=Counter(counts)
    frequency.append(counting)
    plt.plot(edges[:-1],counts)
    plt.title('Histogram of time averaged data over interval={interval} for channel {channel}'.format(interval=interval,channel=index+1))
    plt.xlabel('Count')
    plt.ylabel('Frequency')
    plt.grid()
    plt.show()
    
#%%    

'''
Investigating correlation between raw counts of channel 1 and 2 
Via Spearman Rank Correlation - expect value -1<s<1
'''
data_to_correlate = histogram_data
corr, _ = spearmanr(data_to_correlate[0],data_to_correlate[1])
print('Correlation = ',corr)
if corr<0.01:
    print('No correlation between dark count from channels 1 and 2')


#%%

for index, channel in enumerate(averaged_data):
    corrected_channel=[]
    for i in channel:
        corrected_channel.append(i-min(channel))
    y=np.linspace(0,max(corrected_channel),len(corrected_channel[0:100]))
#    plt.scatter(channel[0:10],y)
#    plt.scatter(y,channel[0:100])
    plt.hist(corrected_channel,bins=1000)
    plt.show()
    
    
    data_length = len(corrected_channel)
    window = 10
    i = 0
    sum_data = []
    counter=[]
    while i < data_length - window + 1:
        data_interval = corrected_channel[i : i + window]
        interval_sum = sum(data_interval)
        sum_data.append(interval_sum)
        counting = Counter(data_interval)
        counter.append(counting)
        i += window
        
    data_length=len(corrected_channel)

    x=np.linspace(0,len(sum_data),len(sum_data))
    plt.scatter(x,sum_data)
    plt.show()
    
#%%
'''
Histogram of background count for channels 1 and 2
'''
from scipy import optimize
from scipy.optimize import curve_fit
import matplotlib.cm as cm
    
def gaussian(x,mu,sd,a):
        y =  (a * np.exp(-0.5*( ((x-mu) / sd)**2) ))
        return y

hist_count_data=[]
bins=10000
for index, channel in enumerate(data):
    a,b=np.histogram(channel,bins=bins)
    
    
    counting = Counter(a)
    value=[]
    num=[]
    for c,d in counting.items():
        value.append(c)
        num.append(d)
    hist_count_data.append([value,num])
    plt.scatter(value,num,marker='.',color=cm.winter(index/2.0),label='Channel {}'.format(index+1))
plt.title('Histogram with {bins} bins'.format(bins=bins))
plt.xlabel('Number of counts')
plt.ylabel('Frequency')
plt.grid()
plt.legend()
plt.show()

for index,hist in enumerate(hist_count_data):
    max_val=max(hist[1])
    normalised_height=[i/max(hist[1]) for i in hist[1]]
    plt.scatter(hist[0],normalised_height,marker='.',color=cm.winter(index/2.0),label='Channel {}'.format(index+1))
    
    print('Peak = ',max(hist[1]))
    mean=np.mean(hist[0])
    print('Mean = ',mean)
    std=np.sqrt(np.std(hist[0]))
    print('Standard Deviation = ',std)
    x=np.linspace(0,500)
    plt.plot(x,gaussian(x,mean,std,1),':',color='black',label='Gaussian Fit')
    
    plt.title('Histogram for channel {channel} with {bins} bins'.format(channel=index+1,bins=bins))
    plt.xlabel('Number of counts')
    plt.ylabel('Frequency')
    plt.grid()
    plt.legend()
    plt.show()

'''
Distribution mean = 20.6 | 40.5
Standard Deviation = 3.16 | 3.67
Distribtuion Peak = 907 | 640
'''
#%%
'''
Dissociating background counts from signal data
'''

'''
Background Data Stats

Channel 1 | Channel 2
Number of events = 191543 | 391110
Mean Count = 0.0064 | 0.013
Correlation = 0.007 = NO correlation
Distribution mean = 22.1 | 41.5
Standard Deviation = 3.03 | 3.72
Distribtuion Peak = 949 | 677
'''

    
means=np.array([[22.1,41.5],[20.6,40.5]])
stds=np.array([[3.03,3.72],[3.16,3.67]])
peaks=np.array([[949,677],[907.640]])

for index, channel in enumerate(hist_count_data):
    max_val=max(channel[1])
    normalised_height=[i/max(channel[1]) for i in channel[1]]
    plt.scatter(channel[0],channel[1],marker='.',color=cm.winter(index/2.0),label='Data channel {}'.format(index+1))

    x=np.linspace(0,500)
    plt.plot(x,gaussian(x,means[0][index],stds[0][index],peaks[0][index]),':',color='black',label='Background Distribution')
    plt.title('Channel {channel}'.format(channel=index+1))
    plt.grid()
    plt.legend()
    plt.xlabel('Number of counts')
    plt.ylabel('Frequency')
    plt.savefig('output/background+signal_distribution.eps')
    plt.show()
    

#%%

'''
Data Collection Stats:
Sample time = 30s
Pulse width = 10microseconds = 10^-5 seconds = 10^7 picoseconds
Pulse frequency = 1Hz = one pulse every 1 second = 10^12 picoseconds
Vpp = 4V
'''
'''
Using 2s sample
Expect 2 pulses
'''
time_crop = 7.5
crop_background=np.array([background[0][:int(len(background[0])/time_crop)] , background[1][:int(len(background[1])/time_crop)]])
crop_data=np.array([data[0][:int(len(data[0])/time_crop)] , data[1][:int(len(data[1])/time_crop)]])

print('Data Interval = {time} seconds'.format(time=30/time_crop))

adjust_background= np.array([[i-min(crop_background[0]) for i in crop_background[0]],[i-min(crop_background[1]) for i in crop_background[1]]])
adjust_data= np.array([[i-min(crop_data[0]) for i in crop_data[0]],[i-min(crop_data[1]) for i in crop_data[1]]])

for index,channel in enumerate(adjust_data):

    print('Data Interval = {time} seconds'.format(time=30/time_crop))
    
    b_detections = len(adjust_background[index])
    detections = len(channel)
    net_detections = int(detections) - int(b_detections)
    print('{det} background detections for channel {channel}'.format(det=b_detections,channel=index+1))
    print('{det} detections for channel {channel}'.format(det=detections,channel=index+1))
    print('{det} net detections for channel {channel}'.format(det=net_detections,channel=index+1))

    b=np.zeros(int(len(adjust_background[index])))
    plt.scatter(adjust_background[index],b,color=cm.autumn(int(index)/2.0),label='Background')
    
    y=np.full(int(len(channel)),index+1)
    plt.scatter(channel,y,color=cm.winter(int(index)/2.0),label='Channel {channel}'.format(channel=int(index)+1))
        

plt.legend()
plt.xlabel('Time Tag')
plt.ylabel('Channel')
plt.grid(axis = 'x')
plt.xlim(0,1E10)
plt.show()
#%%

for i,value in enumerate(adjust_data):
    b1,b2=np.histogram(adjust_background[i],bins=1000)
    s1,s2=np.histogram(value,bins=1000)
    
#    plt.hist(adjust_background[i],bins=100,color='red')
#    plt.hist(data,bins=100,color='green')
    plt.bar(s2[:-1],s1,width=1E12,color=cm.winter(int(i)/2.0))
    plt.bar(b2[:-1],b1,width=1E12,color='red')
plt.title('Channel {channel}'.format(channel=index+1))
plt.show()

