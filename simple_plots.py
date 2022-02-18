#!/usr/bin/env python3
##%%%%%%%%%%%%%

import numpy as np
import matplotlib.pyplot as plt
import scipy as sp
import scipy.stats
from scipy.stats import spearmanr



# tags_channel_list = np.load('output/tags_channel_list_11022022_14_38_42.npy')
# collected_tags = np.load('output/collected_tags_11022022_14_38_42.npy')

background_tags = np.load('data/collected_tags_15022022_16_55_30.npy')
tags_channel_list = np.load('data/tags_channel_list_15022022_16_55_30.npy')

print(background_tags)

print(tags_channel_list)

print(len(background_tags))
##%%%%%%%%%%%%%
channel1 = []
channel2 = []
channel3 = []
channel4 = []

for index, tag in enumerate(background_tags):

    if tags_channel_list[index] == 1.:
        channel1.append(tag)

    if tags_channel_list[index] == 2.:
        channel2.append(tag)
    #
    # if tags_channel_list[index] == 3.:
    #     channel3.append(tag)
    #
    # if tags_channel_list[index] == 4.:
    #     channel4.append(tag)


y_ch1 = np.ones(len(channel1[:1000000]))
y_ch2 = np.ones(len(channel2[:1000000]))
# y_ch3 = np.ones(len(channel3))
# y_ch4 = np.ones(len(channel4))

# plt.scatter(channel1[:1000000], y_ch1)
#
# plt.show()
# plt.scatter(channel2[:1000000], y_ch2)
#
# plt.show()


data = [channel1,channel2]

histogram_data=[]
bins=[3E3,3E4,3E5,3E6,3E7,3E8]
correlations=[]
for bin in bins:
    histogram_ind=[]
    for index, channel in enumerate(data):
        time = max(channel) - min(channel)
        tags_no = len(channel)
        print('Number of events = ',tags_no)

        counts, edges = np.histogram(channel,bins = int(bin))
        print('Mean count = ',np.mean(counts))
        histogram_ind.append(counts)


        interval = 10000
        i = 0
        average_channel = []
        while i < tags_no - interval + 1:
            data_interval = channel[i : i + interval]
            interval_average = sum(data_interval) / int(interval)
            average_channel.append(interval_average)
            i += interval

        counts, edges = np.histogram(average_channel,bins = int(3E5))

    histogram_data.append(histogram_ind)
    corr, _ = spearmanr(histogram_ind[0],histogram_ind[1])
    correlations.append(corr)
plt.plot(bins, correlations)
plt.show()

    # plt.plot(edges[:-1],counts)
    # plt.show()


    # plt.plot(edges[:-1],counts,label='Channel{}'.format(index+1))
    # plt.legend(loc = 'best')
    # plt.title('Background Histogram')
    # plt.show()


corr, _ = spearmanr(histogram_data[0][0],histogram_data[0][1])
print(corr)
if corr<0.01:
    print('No correlation between dark count from channels 1 and 2')
