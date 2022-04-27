#!/usr/bin/env python3
#%%

import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('../')
from subroutines import retrodict_subroutine as probby
from subroutines import curvefit_subroutine as oddy
from subroutines import retrofit_macroroutine as rexxy

fig = plt.figure()

clicks = np.array([67, 360, 858, 867, 336])
clicks = np.array([75.,  583., 1278.,  486.,   63.])
# clicks = np.array([ 40., 241., 552., 772., 611., 228.,  39.,   4., 0.])
# # clicks = np.array([6.,  48., 212., 598., 798., 592., 216.,  18.,   1.])
# # clicks = np.array([  6.,  44., 214., 527., 783., 595., 267.,  51.,   2.])

clicks = [104., 355., 610., 586., 471., 230., 75., 36., 18., 0., 0., 0., 0., 0., 0., 0., 0.]
clicks = [ 20., 100., 265., 500., 547., 456., 335., 168.,  72.,  21., 3., 1., 0., 0., 0., 0., 0.]

clicks = [ 19., 102., 268., 503., 565., 444., 332., 172.,  63.,  15.,   4.,   1.,   0.,   0., 0.,   0.,   0.]
clicks = [  0.,   0.,   7.,  13.,  43., 129., 248., 427., 501., 465., 343., 194.,  81.,  36., 2.,   1.,   0.]

clicks_prob = clicks/np.sum(clicks)
#############################THIS IS THE MAXIMUM LIKELIHOOD ESTIMATOR TEST
multiplex = 16
noise = 0.0001
qe = 0.589

x_domain = np.arange(0, multiplex +1)
mean_guess = np.sum(x_domain * np.array(clicks_prob)) 
print(mean_guess)

output = []
trials = np.linspace(np.floor(mean_guess/3),np.floor(mean_guess*3.5), round(mean_guess*3.5))
print(trials)

print('hihihi',probby.log_mle_pc([9.5, noise, multiplex, qe], x_domain, clicks))

fits = rexxy.poisson_mle_gradient(clicks, multiplex, filename = 'pipeline5_test', start_trial = mean_guess, qe = 0.589, noise = 0.0001, threshold = 0.0001)

print(fits)


fit = probby.noisy_poisson_pc([fits[0], noise, multiplex, qe], x_domain)

# lsq = np.average((fit - clicks_prob)**2)
# print('least squares', lsq)

# fit2 = probby.noisy_poisson_pc([8.857, noise, multiplex, qe], x_domain)
# lsq2 = np.average((fit2 - clicks_prob)**2)
# print('least squares', lsq2)



plt.plot(fit)
plt.plot(clicks_prob)
plt.show()

#%%

##### manual data monkey stuff.....fitting each  

'''
format => eg photon4_50k = [multiplex, qe, noise, [individual count value array]]

'''
qe = 0.589

photon4_50k = [4, qe, 0.006563, [508., 1082., 663., 215., 15.]]
photon4_70k = [4, qe, 0.006385, [78., 536., 1179., 615., 77.]] 
photon4_150k = [4, qe, 0.027358, [5., 87., 865., 1223., 309.]]

photon8_60k = [8, qe, 0.004672, [234., 661., 864., 493., 186., 37., 7., 1., 0.]]
photon8_100k = [8, qe, 0.008244, [40., 232., 522., 775., 610., 239., 63., 5., 1.]]
photon8_140k = [8, qe, 0.013272, [6., 44., 214., 527., 783., 595., 267., 51., 2.]]

photon16_50k = [16, qe, 0.00635, [493., 830., 673., 344., 113., 30., 3., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
photon16_100k = [16, qe, 0.01136, [104., 355., 610., 586., 471., 230., 75., 36., 18., 0., 0., 0., 0., 0., 0., 0., 0.]]
photon16_150k = [16, qe, 0.01725, [19., 102., 268., 503., 565., 444., 332., 172., 63., 15., 4., 1., 0., 0., 0., 0., 0.]]
photon16_200k = [16, qe, 0.0246, [1., 25., 119., 284., 452., 508., 472., 355., 182., 66., 16., 8., 1., 0., 0., 0., 0.]]
photon16_250k = [16, qe, 0.03203, [1., 5., 15., 74., 224., 372., 546., 506., 398., 217., 91., 34., 6., 1., 0., 0., 0.]]
photon16_300k = [16, qe, 0.03555, [0., 0., 6., 11., 60., 128., 286., 455., 536., 462., 307., 165., 63., 11., 0., 0., 0.]]

data = [photon4_50k, photon4_70k, photon4_150k, photon8_60k, photon8_100k, photon8_140k, photon16_50k, photon16_100k, photon16_150k, photon16_200k, photon16_250k, photon16_300k]

for subdata in data:

    multiplex, qe, noise, counts = subdata
    fits = rexxy.poisson_mle_gradient(counts, multiplex, filename = 'pipeline5_test', qe = qe, noise = noise, threshold = 0.0001)


######################################################### THIS IS THE SCIPY ODR FIT TEST
#%%
# noise = 0.01
# efficiency = 0.95
# detector_no = 16
# x = np.arange(0, detector_no+1)
# print(x)
# # output = probby.noisy_pcn(x, 0.1, 3)
# # hi = probby.noisy_poisson_pc(3)
# fit_results = oddy.odrfit(probby.noisy_poisson_pc, x, clicks_prob, [15., noise, detector_no, efficiency])
# plt.figure(figsize=(10, 10))
# plt.plot(clicks_prob,'--',color='red',label='Observed Photon16 Click Distribution')

# output = probby.noisy_poisson_pc(fit_results[0], x)
# plt.plot(output, '--', label = 'Fit Poissonian light distribution, {} mean'.format(fit_results[0][0]))

# for i in range(1, 15, 3):
#     print(i)
#     output = probby.noisy_poisson_pc([float(i)], x)
#     plt.plot(output, label = 'Modelled Poissonian Retrodict, {}.0 mean'.format(i))
# plt.legend()
# plt.xlabel('Detector Click Count')
# plt.ylabel('Normalised Relative Probability')
# plt.title('Raw Observation versus pure Fock States in retrodict model')

# plt.savefig('../output/datavspoisson2.png', dpi = 200)
# plt.show()



#%%
