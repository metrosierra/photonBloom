#!/usr/bin/env python3
#%%

import numpy as np
import matplotlib.pyplot as plt

import sys 
sys.path.append('../')
from subroutines import retrodict_subroutine as probby
from subroutines import curvefit_subroutine as oddy
from subroutines import retrofit_macroroutine as rexy

# fig = plt.figure()

# clicks = np.array([67, 360, 858, 867, 336])
# clicks = np.array([75.,  583., 1278.,  486.,   63.])
# # clicks = np.array([ 40., 241., 552., 772., 611., 228.,  39.,   4., 0.])
# # # clicks = np.array([6.,  48., 212., 598., 798., 592., 216.,  18.,   1.])
# # # clicks = np.array([  6.,  44., 214., 527., 783., 595., 267.,  51.,   2.])

# clicks = [104., 355., 610., 586., 471., 230., 75., 36., 18., 0., 0., 0., 0., 0., 0., 0., 0.]
# clicks = [ 20., 100., 265., 500., 547., 456., 335., 168.,  72.,  21., 3., 1., 0., 0., 0., 0., 0.]

# clicks = [ 19., 102., 268., 503., 565., 444., 332., 172.,  63.,  15.,   4.,   1.,   0.,   0., 0.,   0.,   0.]
# clicks = [  0.,   0.,   7.,  13.,  43., 129., 248., 427., 501., 465., 343., 194.,  81.,  36., 2.,   1.,   0.]

# clicks_prob = clicks/np.sum(clicks)
# #############################THIS IS THE MAXIMUM LIKELIHOOD ESTIMATOR TEST
# multiplex = 16
# noise = 0.0001
# qe = 0.589

# x_domain = np.arange(0, multiplex +1)
# mean_guess = np.sum(x_domain * np.array(clicks_prob)) 
# print(mean_guess)

# output = []
# trials = np.linspace(np.floor(mean_guess/3),np.floor(mean_guess*3.5), round(mean_guess*3.5))
# print(trials)

# print('hihihi',probby.log_mle_pc([9.5, noise, multiplex, qe], x_domain, clicks))

# fits = rexxy.poisson_mle_gradient(clicks, multiplex, filename = 'pipeline5_test', start_trial = mean_guess, qe = 0.589, noise = 0.0001, threshold = 0.0001)

# print(fits)


# fit = probby.noisy_poisson_pc([fits[0], noise, multiplex, qe], x_domain)

# # lsq = np.average((fit - clicks_prob)**2)
# # print('least squares', lsq)

# # fit2 = probby.noisy_poisson_pc([8.857, noise, multiplex, qe], x_domain)
# # lsq2 = np.average((fit2 - clicks_prob)**2)
# # print('least squares', lsq2)



# plt.plot(fit)
# plt.plot(clicks_prob)
# plt.show()

#%%

##### manual data monkey stuff.....fitting each  

'''
format => eg photon4_50k = [multiplex, qe, noise, [individual count value array]]

'''
qe = 0.589

data = {
'photon4_20k': [4, qe, 0.0005436, [1.183e+03, 9.550e+02, 2.940e+02, 3.500e+01, 1.000e+00]],
'photon4_30k': [4, qe, 0.001107325, [769., 1054., 524., 127., 8.]],

'photon4_50k': [4, qe, 0.0032, [546., 1022.,  668.,  223., 24.]],
'photon4_70k': [4, qe, 0.0031, [83., 438., 894., 822., 248.]], 
'photon4_150k': [4, qe, 0.0136, [5., 56., 354., 1051., 1023.]],

'photon8_60k': [8, qe, 0.0031538, [258., 634., 850., 487., 202.,  40.,  10.,   2.,   0.]],
'photon8_100k': [8, qe, 0.005564, [47., 240., 491., 694., 586., 316., 95., 16., 2.]],
'photon8_140k': [8, qe, 0.008958, [6.,  53., 176., 441., 663., 637., 364., 132.,  17.]],

'photon16_50k': [16, qe, 0.0049, [510., 845., 645., 336., 116., 33., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
'photon16_100k': [16, qe, 0.0088, [111., 361., 595., 587., 454., 237., 86., 32., 20., 2., 0., 0., 0., 0., 0., 0., 0.]],
'photon16_150k': [16, qe, 0.0133, [21., 102., 269., 494., 538., 446., 336., 171., 80., 24., 6., 1., 0., 0., 0., 0., 0.]],
'photon16_200k': [16, qe, 0.0190, [2., 27., 118., 288., 397., 512., 450., 365., 196., 95., 27., 10., 1., 1., 0., 0., 0.]],

'photon16_250k': [16, qe, 0.0248, [1., 4., 20., 62., 196., 357., 457., 524., 402., 265., 139., 44., 17., 2., 0., 0., 0.]],
'photon16_300k': [16, qe, 0.0275, [0., 0., 7., 11., 43., 113., 225., 36., 458., 481., 381., 232., 111., 45., 5., 2., 0.]],
'photon16_350k': [16, qe, 0.0309, [0., 0., 0., 2., 8., 18., 48., 129., 270., 417., 513., 438., 367., 185., 79., 15., 1.]],
'photon16_400k': [16, qe, 0.0380, [0., 0., 0., 0., 1., 7., 19., 47., 122., 222., 422., 481., 515., 384., 195., 64., 11.]],
'photon16_450k': [16, qe, 0.0462, [0., 0., 0., 0., 0., 0., 1., 8., 31., 92., 205., 421., 537., 573., 412., 176., 34.]],
'photon16_500k': [16, qe, 0.0544, [0., 0., 0., 0., 0., 0., 2., 2., 17., 39., 102., 282., 485., 593., 567., 326., 75.]]
}

for key in data.keys():

    multiplex, qe, noise, counts = data[key]
    fits = rexy.poisson_mle_gradient(counts, multiplex, filename = key, qe = qe, noise = noise, threshold = 0.0001)
    print(key, fits)

########################################################


# multiplex = 16
# x_domain = np.arange(0, 17)

# opt_fit1 = probby.noisy_poisson_pc([2.6995656180619636, data['photon16_50k'][2], 16, qe], x_domain)
# print(opt_fit1)
# opt_fit2 = probby.noisy_poisson_pc([5.550382802837447, data['photon16_100k'][2], 16, qe], x_domain)
# print(opt_fit2)
# opt_fit3 = probby.noisy_poisson_pc([8.873452644838695, data['photon16_150k'][2], 16, qe], x_domain)
# print(opt_fit3)
# opt_fit4 = probby.noisy_poisson_pc([11.8354751252511, data['photon16_200k'][2], 16, qe], x_domain)
# print(opt_fit4)








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
