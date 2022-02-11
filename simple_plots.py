#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt



tags_channel_list = np.load('output/tags_channel_list_11022022_14_38_42.npy')
collected_tags = np.load('output/collected_tags_11022022_14_38_42.npy')

print(collected_tags)

channel1 = []
channel2 = []
channel3 = []
channel4 = []

for index, tag in enumerate(collected_tags):

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

plt.scatter(channel1[:1000000], y_ch1)
plt.scatter(channel2[:1000000], y_ch2)

plt.show()
