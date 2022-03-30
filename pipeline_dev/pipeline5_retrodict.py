#!/usr/bin/env python3

import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('../')
from subroutines import retrodict_subroutine as probby
from subroutines import curvefit_subroutine as oddy

fig = plt.figure()

clicks = np.array([67, 360, 858, 867, 336])
# clicks = np.array([75.,  583., 1278.,  486.,   63.])
# clicks = np.array([ 40., 241., 552., 772., 611., 228.,  39.,   4., 0.])
# clicks = np.array([6.,  48., 212., 598., 798., 592., 216.,  18.,   1.])
# clicks = np.array([  6.,  44., 214., 527., 783., 595., 267.,  51.,   2.])
# clicks = [ 20., 100., 265., 500., 547., 456., 335., 168.,  72.,  21., 3., 1., 0., 0., 0., 0., 0.]

clicks = [  0.,   0.,   7.,  13.,  43., 129., 248., 427., 501., 465., 343., 194.,  81.,  36., 2.,   1.,   0.]

clicks_prob = clicks/np.sum(clicks)



noise = 0.01
efficiency = 0.95
detector_no = 16
x = np.arange(0, detector_no+1)
print(x)
# output = probby.noisy_pcn(x, 0.1, 3)
# hi = probby.noisy_poisson_pc(3)
fit_results = oddy.odrfit(probby.noisy_poisson_pc, x, clicks_prob, [15., noise, detector_no, efficiency])
plt.figure(figsize=(10, 10))
plt.plot(clicks_prob,'--',color='red',label='Observed Photon16 Click Distribution')

output = probby.noisy_poisson_pc(fit_results[0], x)
plt.plot(output, '--', label = 'Fit Poissonian light distribution, {} mean'.format(fit_results[0][0]))

for i in range(1, 15, 3):
    print(i)
    output = probby.noisy_poisson_pc([float(i)], x)
    plt.plot(output, label = 'Modelled Poissonian Retrodict, {}.0 mean'.format(i))
plt.legend()
plt.xlabel('Detector Click Count')
plt.ylabel('Normalised Relative Probability')
plt.title('Raw Observation versus pure Fock States in retrodict model')

plt.savefig('../output/datavspoisson2.png', dpi = 200)
plt.show()






# plt.plot(clicks_prob,'--',color='red',label='clicks_prob')

# plt.plot(probby.noisy_poisson_pc(fit_results[0], x), color='blue',label='noisy_poisson_pc')
# plt.show()
# print(fit_results[2])
