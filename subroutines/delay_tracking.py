#!/usr/bin/env python3
#%%
import numpy as np
from itertools import chain
import os
from natsort import natsorted
import subroutines.mathematics as mathy


# '''
# Load data
# '''

# data_dir = 'data/'
# folders = natsorted(os.listdir(data_dir))
# try: 
#     folders.remove('archive')
# except:
#     pass
# print(folders)

# dossier = folders[1]
# files = natsorted(os.listdir(data_dir + dossier))

# tags = np.load(data_dir + dossier + '/' + files[0])
# tags_channel_list = np.load(data_dir + dossier + '/' + files[1])


# channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags, tags_channel_list)


# data = [channel1,channel2]



def data_crop(time_interval,data):
    
    '''
    Takes data slice over selected time interval in seconds.
    '''

    crop_data=[]
    for index, channel in enumerate(data):
        start=min(chain(data[0],data[1]))
        crop_data.append([int(i) for i in channel if int(i) <= int(time_interval+start)])
        
    
    max_time = max(chain(crop_data[0],crop_data[1]))
    min_time = min(chain(crop_data[0],crop_data[1]))
    net_time = max_time-min_time
    print('Sample time interval = {t} seconds'.format(t=net_time*1E-12))
    
    for index, channel in enumerate(crop_data):
        print('{N} signals detected in channel {channel} in {t} seconds.'.format(N=len(channel),channel=index+1,t=net_time*1E-12))
        if min(channel) == min_time:
            print('First detection in channel {channel} with timestamp t={t} ps.'.format(channel=index+1,t=start))
        if max(channel) == max_time:
            print('Last detection in channel {channel} after time t={t} s.'.format(channel=index+1, t=net_time*1E-12))
    
    return np.array(crop_data)

def pulse_delay(photonN, delay_lengths, pulse_width, pulse_frequency):
    '''
    Calculates pulse delay times and lengths for photonN
    '''
    c = 299792458 #m/s
    cg = c/1.5 #m/s
    
    pulse_period = 1/pulse_frequency
    
    print('Period of undelayed pulse = {p} s'.format(p=pulse_period))
    pulse_delay=[L/cg for L in delay_lengths]
    
    for i,L in enumerate(delay_lengths):

        print('Photon{N} pulse delay = {pd} seconds with delay length = {L} m.'.format(N=photonN[i],L=L,pd=pulse_delay[i]))
    
    return pulse_delay

# '''
# Calling functions
# '''

# time_interval = 1E12
# data_crop(time_interval,data)


# photonN=[4,8,16]
# delay_lengths = [70,210,490] #in metres
# pulse_width = 50*1E-9 #in seconds
# pulse_frequency = 1000000 #in Hz

# pulse_delay(photonN, delay_lengths, pulse_width, pulse_frequency)