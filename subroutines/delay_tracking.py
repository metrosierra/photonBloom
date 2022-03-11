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



def data_crop(data, time_interval, verbose = False):
    
    '''
    Takes data slice over selected time interval in seconds.
    '''

    output = []

    for channel in data:

        start = channel[0]
        output.append(channel[channel < time_interval + start])

    
    max_time = max([channel[-1] for channel in output])
    min_time = min([channel[0] for channel in output])
    net_time = max_time - min_time
    print('Sample time interval = {t} seconds'.format(t=net_time*1E-12))
    
    if verbose:
        for index, channel in enumerate(output):
            print('{N} signals detected in channel {channel} in {t} seconds.'.format(N=len(channel),channel=index+1,t=net_time*1E-12))
            if min(channel) == min_time:
                print('First detection in channel {channel} with timestamp t={t} ps.'.format(channel=index+1,t=start))
            if max(channel) == max_time:
                print('Last detection in channel {channel} after time t={t} s.'.format(channel=index+1, t=net_time*1E-12))
        
    return output


def channel_chop(channel_data, chop_time):

    offset = channel_data[0]
    totaltime = channel_data[-1] - offset
    print(totaltime)
    chops_no = round(totaltime / chop_time)
    output = []
    for i in range(chops_no):
        temp = channel_data.copy()
        start = chop_time * i 
        end = chop_time * (i+1)
        temp = temp[temp < end + offset]
        temp = temp[temp > start + offset]
        output.append(temp)

    return output 







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