#!/usr/bin/env python3

import numpy as np
import probabilities as probby
import subroutines.odr_fit as oddy
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit

fig = plt.figure()

clicks = np.array([67, 360, 858, 867, 336])
clicks = np.array([75.,  583., 1278.,  486.,   63.])
clicks = np.array([ 40., 241., 552., 772., 611., 228.,  39.,   4., 0.])
# clicks = np.array([6.,  48., 212., 598., 798., 592., 216.,  18.,   1.])

clicks_prob = clicks/np.sum(clicks)
print('hihi', np.sum(clicks_prob))
plt.plot(clicks_prob,'--',color='red',label='clicks_prob')



noise = 0.01
efficiency = 0.95
detector_no = 8

x = np.arange(0, 9)
print(x)
# output = probby.noisy_pcn(x, 0.1, 3)
# hi = probby.noisy_poisson_pc(3)
# fit_results = oddy.fit(probby.noisy_retrodict, x, clicks_prob, initials = np.array([0.1, 0.8]))
photon_no=np.arange(2,12)

hi = probby.noisy_poisson_pc(4.5, detectors = detector_no, qe = 0.95, noise = 0.01)
print(hi)
plt.plot(x, hi/np.sum(hi))
plt.show()



# for no in photon_no:
#     opt,cov = curve_fit(probby.noisy_pcn, x, clicks_prob, p0 = [no, no/1.])
#     print(opt,' hihiihih')
#     yfit = probby.noisy_pcn(x, *opt)


#     chi2 = np.sum((clicks_prob - yfit)**2)/len(yfit)
#     plt.plot(x, yfit, label='{} photons, {:.2f}noise, chi2 = {}'.format(no, opt[0], chi2))

   
#     # # plt.plot(clicks / np.sum(clicks))

#     # plt.plot(output,label='output')
#     fig.patch.set_facecolor('xkcd:sky blue')

#     plt.title(label = 'Noisy Retrodict Efficiency={e}, Noise = {n}'.format(e=0.95,n=opt[0] ))
#     # plt.savefig('output/Noisy_retrodict_E85_N10.eps')
#     # plt.savefig('output/Noisy_retrodict_E85_N10.png')
#     plt.legend()
# plt.plot(clicks_prob,'--',color='red',label='clicks_prob')
# plt.show()


# def chi_square(data,expected):
#     degree = len(data) - 2
#     chi2 =  np.sum((data-expected)**2) / expected / degree
#     return chi2