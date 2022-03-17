#!/usr/bin/env python3

import numpy as np
import probabilities as probby
import subroutines.odr_fit as oddy
import matplotlib.pyplot as plt

clicks = np.array([67, 360, 858, 867, 336])

noise = 0.01
efficiency = 0.95
detector_no = 4

x = np.arange(0, 5)
print(x)
output = probby.noisy_retrodict([6], x)
print(output)
# fit_results = oddy.fit(probby.noisy_retrodict, x, clicks, initials = np.array([2.]))


# plt.plot(clicks / np.sum(clicks))
plt.plot(output)
