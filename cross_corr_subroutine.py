#%%%
#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt
from numba import njit, prange
from natsort import natsorted
import os

import mathematics as mathy

###test
@njit(parallel = True)
def signal_bin_combing(data, bin_width, sig_bin_no = 1, period_no = 100):

    total_time = np.max(data) - np.min(data)
    # print(total_time/1e12, 'time')
    bin_no = int(np.floor(total_time / bin_width))

    period_cycles = int(np.floor(bin_no / period_no))
    print(period_cycles, 'cycles')
    counts, edges = mathy.numba_histogram(data, bin_no)

    output_data = []
    for i in range(period_no):

        sig_count = 0
        for q in prange(period_cycles-1):

            start_index = q*(period_no-1) + i
            sig_count += np.sum(counts[start_index : start_index + sig_bin_no])

        output_data.append(sig_count/period_cycles)

    return output_data



data_dir = 'data/'
folders = natsorted(os.listdir(data_dir))
try: 
    folders.remove('archive')
except:
    pass
print(folders)

#%%%
###because of natsorted (natural sort), collected_blah.npy should be
###in front of tags_blah.npy because alphabetical order

dossier = folders[0]
files = natsorted(os.listdir(data_dir + dossier))

tags = np.load(data_dir + dossier + '/' + files[0])
tags_channel_list = np.load(data_dir + dossier + '/' + files[1])


channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags, tags_channel_list)

index = 2000
width = 1e4
signo = 100
peno = 200000


time = (np.max(channel1[:index]) - np.min(channel1)) / 1e12
output1 = signal_bin_combing(channel1[:index], bin_width = width, sig_bin_no = signo, period_no = peno)
output2 = signal_bin_combing(channel2[:index], bin_width = width, sig_bin_no = signo, period_no = peno)

output1 = np.array(output1)
output2 = np.array(output2)

plt.plot(output1, label = 'channel1')
plt.legend()
plt.xlabel('Combing index')
plt.ylabel('Average signal bin count')
plt.xlim([97500, 102500])
plt.savefig('output/channel1_{}_bin{}_sig{}_period{}_seconds{:.3g}.eps'.format(dossier, width, signo, peno, time), bbox = 'tight')
plt.show(block = False)
plt.close()
plt.plot(output2, label = 'channel2')
plt.legend()
plt.xlabel('Combing index')
plt.ylabel('Average signal bin count')
plt.xlim([97500, 102500])
plt.savefig('output/channel2_{}_bin{}_sig{}_period{}_seconds{:.3g}.eps'.format(dossier, width, signo, peno, time), bbox = 'tight')
plt.show(block = False)
plt.close()



# %%
