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


# counts = np.array([67, 360, 858, 867, 336])

# output = rexy.poisson_retrofit(counts, multiplex = 4, filename = 'hihi')

#%%
stages = [4, 8, 16]

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
        file_path = data_dir + folder + '/' + dossiers[0]
        files = natsorted(os.listdir(file_path))
        tags = np.load(file_path + '/' + files[0])
        tags_channel_list = np.load(file_path + '/' + files[1])
        channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags, tags_channel_list, commonfloor = True)
        data = [channel1, channel2]

    ####################################################

############################## workhorse is just here 
        counts = twosie.twomultiplex(data, chopsize = 0.01e12, chop_no = 5, binwidth = 1e3, peno = 20001, multiplex = multiplex, filename = filename)

        output = rexy.poisson_retrofit(counts, multiplex = multiplex, filename = filename)
####################################




#%%