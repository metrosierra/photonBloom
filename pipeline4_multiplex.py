#!/usr/bin/env python3
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import os
from natsort import natsorted

import subroutines.mathematics as mathy
import subroutines.delay_tracking as deli
import subroutines.multiplex_macroroutine as molly


targets = ('1k_countrate_50nsbench/', '10k_countrate_50nsbench/', '50k_countrate_50nsbench/','100k_countrate_50nsbench/')
targets = ('70k_countrate_50nsbench/',)

for target_dir in targets:
    data_dir = 'data/photon4/' + target_dir
    folders = natsorted(os.listdir(data_dir))
    try:
        folders.remove('archive')
    except:
        pass
    print(folders)
    #%%
    ###because of natsorted (natural sort), collected_blah.npy should be
    ###in front of tags_blah.npy because alphabetical order
    for dossier_index in range(2):
        dossier = folders[dossier_index]
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
        data = [channel1, channel2]
        channel1fixed, channel2fixed = deli.data_crop(data, 0.07e12)


        channel1chops = deli.channel_chop(channel1fixed, 0.01e12)
        channel2chops = deli.channel_chop(channel2fixed, 0.01e12)
        # channel1chops = [channel1]
        # print(len(channel1), 'hiiii')
        #%%
        width = 1e3
        signo = 800
        peno = 20001
        multiplex = 2
        chop_no = 5

        output1 = molly.sig_chops_multiplex(channel1chops, chop_no = chop_no, binwidth = width, sig_bin_no = signo, sig_threshold = 1, period_no = peno, multiplex = multiplex)
        output2 = molly.sig_chops_multiplex(channel2chops, chop_no = chop_no, binwidth = width, sig_bin_no = signo, sig_threshold = 1, period_no = peno, multiplex = multiplex)


        print(output1)

        #%%

        ceiling = min([len(output1), len(output2)])
        aggregate = output1[:ceiling] + output2[:ceiling]
        hi3 = plt.hist(aggregate, density = False, bins = np.arange(-0.5, 9.5), label = '{} Data photon{} statistics'.format(target_dir[:-1], multiplex*2))

        print(hi3[0])
        poinput = np.array([0, 1, 2, 3, 4])
        poinput = np.array([0, 1, 2, 3, 4, 5, 6, 7, 8])
        mean = np.dot(poinput, hi3[0]) / np.sum(hi3[0])
        print(mean)

        poisson = np.array([mathy.poissonian(mean, x) for x in poinput])
        plt.plot(poinput, poisson*np.sum(hi3[0]), label = 'Theoretical Poissonian ($\lambda$ = {:.4g})'.format(mean))
        plt.legend()
        plt.title('{}'.format(hi3[0]))
        plt.savefig('output/{}_{}_Data photon{}_statistics.eps'.format(target_dir[:-1], dossier, multiplex*2))
        print(poisson)
        plt.show(block = False)
        plt.close()
