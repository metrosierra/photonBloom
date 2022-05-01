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
from subroutines import retrofit_macroroutine as rexxy


# counts = np.array([67, 360, 858, 867, 336])

# output = rexy.poisson_retrofit(counts, multiplex = 4, filename = 'hihi')

#%%
stages = [4, 8, 16]
# stages = [2]
'''
file structure is just:
data -> photon{stages} -> multiple folders for each countrate -> each countrate folder 
which has a darkcount folder and signal folder (eg 50nswidth_blah)

os.listdir lists the folder names automatically after photonX
'''


qe = 0.589
### delay time in ns, usually 350
delayno = 350
### pulse time in ns, usually 50
pulseno = 20
### whether to use old fitting method or new
old = False


######################just finding the file
for stage in stages:


    data_dir = '../data/photon{}/'.format(stage)
    folders = natsorted(os.listdir(data_dir))
    try:
        folders.remove('archive')
    except:
        pass
    print(folders)
    #%%
    ###because of natsorted (natural sort), collected_blah.npy should be
    ###in front of tags_blah.npy because alphabetical order
    
    for folder in folders: 

        multiplex = stage
        filename = 'photon{}_{}'.format(stage, folder)
    ####### just some file accessing shenanigans
        dossiers = natsorted(os.listdir(data_dir + folder))
        print(dossiers)

        signal_path = data_dir + folder + '/' + dossiers[0]
        signal_files = natsorted(os.listdir(signal_path))
        tags = np.load(signal_path + '/' + signal_files[0])
        tags_channel_list = np.load(signal_path + '/' + signal_files[1])
        channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags, tags_channel_list, commonfloor = True)
        data = [channel1, channel2]

    ####################################################
        noise_path = data_dir + folder + '/' + dossiers[1]
        noise_files = natsorted(os.listdir(noise_path))
        tags = np.load(noise_path + '/' + noise_files[0])
        tags_channel_list = np.load(noise_path + '/' + noise_files[1])
        c1, c2, c3, c4 = mathy.tag_fourchannel_splice(tags, tags_channel_list, commonfloor = True)
        noisedata = [np.array(c1), np.array(c2)]

        ### this is the time of a single multiplexed pulse window (delay+one subpulse) in ns
        windowno = delayno + pulseno
        ###just getting average noise probability per ns 
        noiseprob = 0.
        for noise in noisedata:
            ### let's say we average over the first 8 seconds to be safe
            ### from undercounting...this is all in nanoseconds
            noiserate = len(noise[noise < (noise[0] + 8e12)]) / 8e9
            noiseprob += noiserate * windowno / 2
            print(noiseprob, 'average noise')

    ####################################################


    ############################## workhorse is just here 
        counts = twosie.twomultiplex(data, chopsize = 0.3e12, chop_no = 3, binwidth = 1e3, peno = 20001, multiplex = multiplex, filename = filename, delayno = delayno, pulseno = pulseno)
        # counts = twosie.twomultiplex(data, chopsize = 0.1e12, chop_no = 6, binwidth = 1e3, peno = 20000, multiplex = multiplex, filename = filename, delayno = 0, pulseno = 50)
        ### old
        if old:
            output = rexxy.poisson_odr_retrofit(counts, multiplex = multiplex, filename = filename, qe = qe, noise = noiseprob)
        else:
        ### new MLE gradient descent
            output = rexxy.poisson_mle_gradient(counts, multiplex, filename = filename, qe = qe, noise = noiseprob, threshold = 0.0001)

    ####################################




#%%