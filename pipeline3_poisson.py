#!/usr/bin/env python3
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy import optimize
from scipy.optimize import curve_fit
import scipy as sp
import scipy.special

import os
from natsort import natsorted

import subroutines.mathematics as mathy
import subroutines.delay_tracking as deli
import subroutines.sigbucket_subroutine as siggy

data_dir = 'data/run2/'
folders = natsorted(os.listdir(data_dir))
try:
    folders.remove('archive')
except:
    pass
print(folders)
#%%
###because of natsorted (natural sort), collected_blah.npy should be
###in front of tags_blah.npy because alphabetical order

dossier = folders[2]
files = natsorted(os.listdir(data_dir + dossier))

tags = np.load(data_dir + dossier + '/' + files[0])
tags_channel_list = np.load(data_dir + dossier + '/' + files[1])


channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags, tags_channel_list)
print(np.min(channel1))

min1 = np.min(channel1)
min2 = np.min(channel2)

if min1 < min2:
    channel2 = np.insert(channel2, 0, min1)

elif min2 < min1:
    channel1 = np.insert(channel1, 0, min2)
#%%
channel1fixed = channel1.copy()
channel2fixed = channel2.copy()


#%%

channel1chops = deli.channel_chop(channel1fixed, 0.05e12)
channel2chops = deli.channel_chop(channel2fixed, 0.05e12)
# channel1chops = [channel1]
# print(len(channel1), 'hiiii')

width = 10e3
signo = 100
peno = 20001

bucket = []


total_events = 0
for i in range(10):

    events1 = len(channel1chops[0])
    events2 = len(channel2chops[0])
    print(events1, 'expected number of events1')
    print(events2, 'expected number of events2')

    total_events += events1 + events2
    length = channel1chops[i][-1] - channel1chops[i][0]
 
    print('Chop', i)
    output1 = siggy.signal_bin_combing(channel1chops[i], bin_width = width, sig_bin_no = signo, period_no = peno)
    shift1 = np.where(output1 == np.max(output1))[0][0]
    print(shift1, 'shift1!!')
    output2 = siggy.signal_bin_combing(channel2chops[i], bin_width = width, sig_bin_no = signo, period_no = peno)
    shift2 = np.where(output2 == np.max(output2))[0][0]
    print(shift2, 'shift2!!')


    experiments1, is_signal1 = siggy.signal_bucketing(channel1chops[i], bin_width = width, sig_bin_no = signo, period_no = peno, index_offset = shift1, signal_threshold = 1)
    experiments2, is_signal2 = siggy.signal_bucketing(channel2chops[i], bin_width = width, sig_bin_no = signo, period_no = peno, index_offset = shift2, signal_threshold = 1)

    ceiling = min([len(is_signal1), len(is_signal2)])
    aggregate = is_signal1[:ceiling] + is_signal2[:ceiling] 
    bucket += [np.sum(run) for run in aggregate]

#     # hi1 = plt.hist(experiments1)
#     # plt.show()
#     # print(hi1[0])
#     # hi2 = plt.hist(experiments2)
#     # plt.show()
#     # print(hi2[0])

    # plt.plot(output1, label = 'channel1')
    # plt.xlabel('Combing index')
    # plt.ylabel('Average signal bin count')

    # plt.show()

#     # plt.plot(output2, label = 'channel2')
#     # plt.xlabel('Combing index')
#     # plt.ylabel('Average signal bin count')

#     # plt.show()

print(total_events)
hi3 = plt.hist(bucket)
plt.show()
print(hi3[0])
# %%

def poisson(x,mean,a):
    xfact = sp.special.factorial(x)
    
    return a * ( (mean**x) / xfact ) * np.exp(-mean)


def poisson_histogram_fit(bucket,savefig=False):
    '''
    Plotting the histogram of the data buckets and fitting with a Poisson distribution.
    '''
    
    hist = plt.hist(np.concatenate(bucket).ravel(),color='grey')

    x=np.array(list(range(3)))
    y = [i for i in hist[0] if int(i) != 0]
    
    mean=np.mean(y)
    print('Mean value = {m}'.format(m=mean))
    
    for j, val in enumerate(y):
        p_val = poisson(j,val,1)
        print('Poisson coefficient for bin {n} = {p}.'.format(n=int(j),p=p_val))
    
    plt.hist(np.concatenate(bucket).ravel(),color='grey')
    plt.scatter(x,y,marker='.',color='red',)
    
    opt, cov = curve_fit(poisson,x,y,maxfev=10000)
    
    xval=np.linspace(0,3)
    plt.plot(xval,poisson(xval,*opt),'--',color='red',label='Poisson Fit\n mean={m}\n a={s}'.format(m=int(opt[0]),s=int(opt[1])))
    
    plt.ylabel('Counts')
    plt.xlabel('Number')
    plt.legend()
    plt.ylim(0,5000)
    
    plt.title('Poisson Fit: 10kcps_50ns_50kHz')
    if savefig is True:
        plt.savefig('output/poissonfit_10kcps_50ns_50kHz.eps')
    plt.show()
    
    fit_mean = opt[0]
    fit_scale = opt[1]
    print('Poisson Fit Parameters:\n Mean = {m} +/- {me} \n Scale = {a} +/- {ae}'.format(m=opt[0],a=opt[1],me=cov[0,0],ae=cov[1,1]))
    
    return fit_mean, fit_scale
    
poisson_histogram_fit(bucket,savefig=False)
