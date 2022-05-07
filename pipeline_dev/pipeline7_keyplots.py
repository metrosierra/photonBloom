#!/usr/bin/env python3
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks


import os
from natsort import natsorted

import sys 
sys.path.append('../')
from subroutines import prettyplot as pretty
from subroutines import retrodict_subroutine as probby
from subroutines import mathematics as mathy
from subroutines import twomultiplex_macroroutine as twosie
from subroutines import retrofit_macroroutine as rexy
from subroutines import cross_corr_subroutine as cross
from subroutines import prettyplot as plot
from subroutines import delay_tracking as deli


#%%
'''
We should treat this script as a jupyter notebook style, using vscode's cell+terminal functionality
to run the code. 

All data henceforth should be hard-coded into the script, with the methods/parameters of how they were
obtained being described in the comments explicitly and completely.

The plotting protocol should be called from prettyplots.py, with allowance for the aspect ratio of the plot, 
otherwise the formatting for all plots should be consistent.

We will stick to .eps format, and the output should be saved to output/keyplots, with appropriate relative 
file paths. 

If we want to use different data, we should state the different parameters in comments above the hardcoded array/list
(straightforward examples below). The point is to separate the data processing script (takes very long) and the plotting script (fast).
This makes the final report crunch much faster.

'''

#%%

'''

keyplots:

Sabrina:
photon8 data only (showing saturation) -> 60k, 100k, 140k 
photon4 data only (showing saturation) -> 20k, 70k, 150k
photon16 bidirectional histogram (using all 10s of data)-> 150k --> pipeline2_correlation_plot.py --> with function cross.cross_corr

Mingsong:
photon16 data + fit -> 50k, 100k, 150k, 200k
photon16 combing ladder -> 150k -> pipeline4_multiplex.py
photon16 jitter smear plot (varying size of data chop) -> 150k -> pipeline4_multiplex.py 



'''

'''
- square plots generally look nice (9x9 plot vs fontsize 20 scaling)
- use inkscape or something to combine eps plots
- inward pointing ticks, for all four sides
- use different linestyle and colour for each line in plot
- use axis labels 'Counts' vs 'Clicks'
- significants figures/decimals reflect precision of quantity measured. else default to 3 sig fig (eg probability)
- errors are reported to ONE SIG FIG, value rounded to error precision
- legends go to largest empty space 
- legends: 'Click Distribution' and 'Poissonian Fit, mean'



'''



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 'photon16_50k': [16, qe, 0.0049, [510., 845., 645., 336., 116., 33., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]],
# 'photon16_100k': [16, qe, 0.0088, [111., 361., 595., 587., 454., 237., 86., 32., 20., 2., 0., 0., 0., 0., 0., 0., 0.]],
# 'photon16_150k': [16, qe, 0.0133, [21., 102., 269., 494., 538., 446., 336., 171., 80., 24., 6., 1., 0., 0., 0., 0., 0.]],
# 'photon16_200k': [16, qe, 0.0190, [2., 27., 118., 288., 397., 512., 450., 365., 196., 95., 27., 10., 1., 1., 0., 0., 0.]],

# 'photon16_250k': [16, qe, 0.0248, [1., 4., 20., 62., 196., 357., 457., 524., 402., 265., 139., 44., 17., 2., 0., 0., 0.]],
# 'photon16_300k': [16, qe, 0.0275, [0., 0., 7., 11., 43., 113., 225., 36., 458., 481., 381., 232., 111., 45., 5., 2., 0.]],
# 'photon16_350k': [16, qe, 0.0309, [0., 0., 0., 2., 8., 18., 48., 129., 270., 417., 513., 438., 367., 185., 79., 15., 1.]],
# 'photon16_400k': [16, qe, 0.0380, [0., 0., 0., 0., 1., 7., 19., 47., 122., 222., 422., 481., 515., 384., 195., 64., 11.]],
# 'photon16_450k': [16, qe, 0.0462, [0., 0., 0., 0., 0., 0., 1., 8., 31., 92., 205., 421., 537., 573., 412., 176., 34.]],
# 'photon16_500k': [16, qe, 0.0544, [0., 0., 0., 0., 0., 0., 2., 2., 17., 39., 102., 282., 485., 593., 567., 326., 75.]]

overall_qe = 0.589###AS OF 26 APRIL 2022 

#%%%%%%%

'''
oscilloscope trace plot for InGaS photodetector setup

The pulse looks like /\ (triangle) on the sig gen as a voltage signal
because 8.4 is shortest rise/fall time for the pulse edges
the oscilloscope trace shows the voltage converted from the PHOTON COUNT

we see a broadening of the trace asymmetrically to a width of 50ns and this is possible due to system
capacitance (takes time to charge up, takes time to discharge according to exponentials)
'''

data = np.loadtxt('../data/F0000CH1.csv')
time = data[:,0]*1e9
voltage = data[:,1]*1e3

fig, ax = pretty.prettyplot(figsize = (9, 9), yaxis_dp = '%.2f', xaxis_dp = '%.0f', ylabel = 'Voltage (mV)', xlabel = 'Time (ns)', title = '')

plt.plot(time, voltage)
plt.xlim([-60, 130])
plt.savefig('../output/laser_pulse_scope_trace.eps')
plt.savefig('../output/laser_pulse_scope_trace.png', dpi = 200)

plt.show()

#%%

'''
Countrate vs fit poisson parameter plot

Where we are taking the total countrate (sum of two channel countrates)
as a linear proxy to the laser power, since we manually adjust the laser
but use the countrate as our "power meter"...

So we assuming the countrate scales linearly, as long as we are well below
individual nanowire saturation

the data sets are photon16->50k, 100k, 150k, 200k, 250k, 300k, 350k, 400k, 450k, 500k, in this order

'''

countrates = np.array([72540.0, 139500.0, 205820.0, 259540.0, 329540.0, 414120.0, 497420.0, 546040.0, 601040.0, 630120.0]
)


poisson_fits = np.array([2.581, 5.212, 8.086, 10.577, 14.328, 19.784, 26.595, 31.919, 38.599, 43.312])
fit_errors = [0.044, 0.064, 0.082, 0.096, 0.116, 0.144, 0.179, 0.209, 0.248, 0.278]

fit1 = np.polyfit(countrates[0:4]/1000, poisson_fits[0:4], 1)

xfit1 = np.linspace(countrates[0]/1000, countrates[4]/1000, 500)
yfit1 = xfit1*fit1[0] + fit1[1]

fit2 = np.polyfit(countrates[4:-1]/1000, poisson_fits[4:-1], 1)

xfit2 = np.linspace(countrates[4]/1000, countrates[-1]/1000, 500)
yfit2 = xfit2*fit2[0] + fit2[1]

fit3 = np.polyfit(countrates/1000, poisson_fits, 2)

xfit3 = np.linspace(countrates[0]/1000, countrates[-1]/1000, 500)
yfit3 = xfit3**2*fit3[0] + xfit3*fit3[1] + fit3[2]

fig, ax = pretty.prettyplot(figsize = (9, 9), yaxis_dp = '%.1f', xaxis_dp = '%.0f', ylabel = 'Poisson Mean Fit', xlabel = 'Count Rate (kHz)', title = '')

plt.plot(np.array(countrates)/1000, poisson_fits, marker = '.', markersize = 18, color = 'black', label = 'Data')

# plt.plot(xfit1, yfit1, color = 'dodgerblue', linewidth = 3, label = 'Low Power Regime Fit')
# plt.plot(xfit2, yfit2, color = 'firebrick', linewidth = 3, label = 'High Power Regime Fit')
plt.plot(xfit3, yfit3, color = 'orange', linewidth = 4, linestyle = '--', label = 'Heuristic Quadratic Fit')
plt.legend(fontsize = 20)

plt.savefig('../output/linearity_study.eps', bbox_inches = 'tight')
plt.savefig('../output/linearity_study.png', dpi = 200)

plt.show()



#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clicks = np.arange(0, 5)
x_axis = np.arange(0, 5)
## 20k photon4

qe = 0.589
left_error = 0.030
right_error = 0.031
fitparam = 1.240
rsquare = 0.9999

data = [4, qe, 0.0005436, [1.183e+03, 9.550e+02, 2.940e+02, 3.500e+01, 1.000e+00]]
photon4_20k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)
fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black', marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon4_20k_fit, color = 'dodgerblue', marker = '.', markersize = 17, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.2f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)


plt.savefig('../output/final_photon4_20k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon4_20k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## 70k photon4

qe = 0.589
left_error = 0.078
right_error = 0.079
fitparam = 5.739
rsquare = 0.998

data = [4, qe, 0.0031, [83., 438., 894., 822., 248.]]
photon4_70k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)

fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black', marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon4_70k_fit, color = 'dodgerblue', marker = '.', markersize = 17, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.2f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)


plt.savefig('../output/final_photon4_70k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon4_70k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## 150k photon4

qe = 0.589
left_error = 0.137
right_error = 0.139
fitparam = 11.0
rsquare = 0.998

data = [4, qe, 0.0136, [5., 56., 354., 1051., 1023.]]
photon4_150k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)

fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black', marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon4_150k_fit, color = 'dodgerblue', marker = '.', markersize = 17, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.1f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)


plt.savefig('../output/final_photon4_150k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon4_150k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clicks = np.arange(0, 9)
x_axis = np.arange(0, 9)
## 60k photon8

qe = 0.589
left_error = 0.055
right_error = 0.055
fitparam = 3.783
rsquare = 0.989

data = [8, qe, 0.0031538, [258., 634., 850., 487., 202.,  40.,  10.,   2.,   0.]]
photon8_60k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)

fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black', marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon8_60k_fit, color = 'dodgerblue', marker = '.', markersize = 17, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.2f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)


plt.savefig('../output/final_photon8_60k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon8_60k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## 100k photon8

qe = 0.589
left_error = 0.079
right_error = 0.078
fitparam = 6.826
rsquare = 0.997  

data = [8, qe, 0.005564, [47., 240., 491., 694., 586., 316., 95., 16., 2.]]
photon8_100k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)

fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black', marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon8_100k_fit, color = 'dodgerblue', marker = '.', markersize = 17, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.2f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)


plt.savefig('../output/final_photon8_100k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon8_100k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
## 140k photon8

qe = 0.589
left_error = 0.104
right_error = 0.105
fitparam = 10.5
rsquare = 0.999  

data = [8, qe, 0.008958, [6.,  53., 176., 441., 663., 637., 364., 132.,  17.]]
photon8_140k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)

fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black', marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon8_140k_fit, color = 'dodgerblue', marker = '.', markersize = 17, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.2f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)


plt.savefig('../output/final_photon8_140k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon8_140k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()


#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clicks = np.arange(0, 17)
x_axis = np.arange(0, 17, 2)

## 50k photon16

qe = 0.589
left_error = 0.044
right_error = 0.044
fitparam = 2.581
rsquare = 0.9994

data = [16, qe, 0.0049, [510., 845., 645., 336., 116., 33., 1., 1., 0., 0., 0., 0., 0., 0., 0., 0., 0.]]
photon16_50k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)

fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black', marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon16_50k_fit, color = 'dodgerblue', marker = '.', markersize = 17, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.2f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)


plt.savefig('../output/final_photon16_50k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon16_50k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### 100k photon16

left_error = 0.064
right_error = 0.065
fitparam = 5.212
rsquare = 0.998

data = [16, qe, 0.0088, [111., 361., 595., 587., 454., 237., 86., 32., 20., 2., 0., 0., 0., 0., 0., 0., 0.]]
photon16_100k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)


fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black', marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon16_100k_fit, color = 'dodgerblue', marker = '.', markersize = 20, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.2f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)

plt.savefig('../output/final_photon16_100k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon16_100k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
### 150k photon16


left_error = 0.082
right_error = 0.083
fitparam = 8.086
rsquare = 0.995
data = [16, qe, 0.0133, [21., 102., 269., 494., 538., 446., 336., 171., 80., 24., 6., 1., 0., 0., 0., 0., 0.]]
photon16_150k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)

fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black', marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon16_150k_fit, color = 'dodgerblue', marker = '.', markersize = 20, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.2f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)

plt.savefig('../output/final_photon16_150k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon16_150k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()

#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
# 200k, photon16

left_error = 0.096
right_error = 0.097
fitparam = 10.577
rsquare = 0.994
data = [16, qe, 0.0190, [2., 27., 118., 288., 397., 512., 450., 365., 196., 95., 27., 10., 1., 1., 0., 0., 0.]]
photon16_200k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)


fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black', marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon16_200k_fit, color = 'dodgerblue', marker = '.', markersize = 20, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.1f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)

plt.savefig('../output/final_photon16_200k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon16_200k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 350k, photon16

left_error = 0.179
right_error = 0.180
fitparam = 26.595
rsquare = 0.996
data = [16, qe, 0.0309, [0., 0., 0., 2., 8., 18., 48., 129., 270., 417., 513., 438., 367., 185., 79., 15., 1.]]
photon16_350k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)


fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black', marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon16_350k_fit, color = 'dodgerblue', marker = '.', markersize = 20, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.1f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)

plt.savefig('../output/final_photon16_350k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon16_350k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()
#%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

# 500k, photon16

left_error = 0.278
right_error = 0.280
fitparam = 43.312
rsquare = 0.998
data = [16, qe, 0.0544, [0., 0., 0., 0., 0., 0., 2., 2., 17., 39., 102., 282., 485., 593., 567., 326., 75.]]
photon16_500k_fit = probby.noisy_poisson_pc_new([fitparam, data[2],  data[0],  data[1]], clicks)

fig, ax = pretty.prettyplot(figsize = (11, 9), yaxis_dp = '%.3f', xaxis_dp = '%.0f', ylabel = 'Probability', xlabel = 'Clicks', title = '')

plt.plot(clicks, data[3]/np.sum(data[3]), color = 'black',marker = 's', markersize = 13, ls = 'solid', linewidth  = 2, label = 'Click Distribution')
plt.plot(clicks, photon16_500k_fit, color = 'dodgerblue', marker = '.', markersize = 20, ls = 'dashed', linewidth = 2, label = '$\lambda_{{MLE}} = {:.1f}\pm{:.1g}, R^2 = {}$'.format(fitparam, right_error, rsquare))
ax.legend(fontsize = 20)
plt.xticks(x_axis)

plt.savefig('../output/final_photon16_500k_mleretrodict_{}percentqe.eps'.format(qe*100))
plt.savefig('../output/final_photon16_500k_mleretrodict_{}percentqe.png'.format(qe*100), dpi = 200)
plt.show(block = False)
plt.pause(1)
plt.close()


#%%
'''
Photon4 data only (showing saturation) -> 20k, 70k, 150k
'''
photon_no = 4
xaxis = np.arange(0, photon_no+1)
'''
Photon4 Data: 20k, 70k, 150k
'''
photon4_20k = [4, overall_qe, 0.0005436, [1.183e+03, 9.550e+02, 2.940e+02, 3.500e+01, 1.000e+00]]
photon4_70k = [4, overall_qe, 0.0031, [83., 438., 894., 822., 248.]]
photon4_150k = [4, overall_qe, 0.0136, [5., 56., 354., 1051., 1023.]]

data_photon4 = [photon4_20k, photon4_70k, photon4_150k]
filename_photon4 = ['photon4_20k_histogram', 'photon4_70k_histogram', 'photon4_150k_histogram']


'''
Histogram Plot
'''

for i, data in enumerate(data_photon4):
    fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.0f', xaxis_dp = '%.0f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
    plt.bar(xaxis, data[3], width=1, color = 'lightsteelblue', edgecolor = 'black',label = 'Click Distribution')
    plt.xticks(xaxis)
    plt.savefig('../output/{}.eps'.format(filename_photon4[i]), bbox_inches = 'tight')
    plt.show()
    plt.close()

'''
Line Plot
'''

for i, data in enumerate(data_photon4):
    fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.1f', xaxis_dp = '%.0f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
    plt.scatter(xaxis, data[3], color = 'black', marker='D',label = 'Click Distribution')
    plt.plot(xaxis, data[3], color = 'black',linestyle='--')
    plt.xticks(xaxis)
    plt.savefig('../output/{}.eps'.format(filename_photon4[i]+'_lineplot'))
    plt.show()
    plt.close()


#%%

'''
Photon8 data only (showing saturation) -> 60k, 100k, 140k
'''
photon_no = 8
xaxis = np.arange(0, photon_no+1)
'''
Photon8 Data: 60k, 100k, 140k
'''
photon8_60k = [8, overall_qe, 0.0031538, [258., 634., 850., 487., 202.,  40.,  10.,   2.,   0.]]
photon8_100k = [8, overall_qe, 0.005564, [47., 240., 491., 694., 586., 316., 95., 16., 2.]]
photon8_140k = [8, overall_qe, 0.008958, [6.,  53., 176., 441., 663., 637., 364., 132.,  17.]]

data_photon8 = [photon8_60k, photon8_100k, photon8_140k]
filename_photon8 = ['photon8_60k_histogram', 'photon8_100k_histogram', 'photon8_140k_histogram']

'''
Histogram Plot
'''

for i, data in enumerate(data_photon8):
    fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.0f', xaxis_dp = '%.0f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
    plt.bar(xaxis, data[3], width=1, color = 'lightsteelblue', edgecolor = 'black',label = 'Click Distribution')
    plt.xticks(xaxis)
    plt.savefig('../output/{}.eps'.format(filename_photon8[i]), bbox_inches ='tight')
    plt.show()
    plt.close()

'''
Line Plot
'''

for i, data in enumerate(data_photon8):
    fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.0f', xaxis_dp = '%.0f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
    plt.scatter(xaxis, data[3], color = 'black', marker='D',label= 'Click Distribution')
    plt.plot(xaxis, data[3], color = 'black',linestyle='--')
    plt.xticks(xaxis)
    plt.savefig('../output/{}.eps'.format(filename_photon8[i]+'_lineplot'), bbox_inches ='tight')
    plt.show()
    plt.close()

#%%
'''
photon16 bidirectional histogram (using all 10s of data)-> 150k --> pipeline2_correlation_plot.py --> with function cross.cross_corr
'''
overall_qe = 0.589

photon_no = 16
xaxis = np.arange(0, photon_no+1)

'''
Loading and Saving Data
'''
tags = np.load('../data/photon16/150k_countrate_50nsbench/50ns_50khz_4vpp2voffset/collected_tags_24032022_17_31_49.npy')
tags_channel_list = np.load('../data/photon16/150k_countrate_50nsbench/50ns_50khz_4vpp2voffset/tags_channel_list_24032022_17_31_49.npy')


channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags, tags_channel_list,commonfloor = True)
data=[channel1,channel2]
channel1, channel2 = deli.data_crop(data, 0.2e12)

print('hihi')
bins = 2000
max_delay = 5e6
counts, midpoints = cross.cross_corr(channel1, channel2, bins = bins, max_delay = max_delay)
midpoints *= 1e-6

bidirectional_data = np.array([midpoints, counts])
np.savetxt('../output/photon8_150k_data_bidirectional.txt', bidirectional_data)
'''
Plotting Bidirectional Histogram Photon16 150k countrate
'''

bidirectional_data = np.loadtxt('../output/photon8_150k_data_bidirectional.txt')

fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.0f', xaxis_dp = '%.2f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
plt.plot(bidirectional_data[0], bidirectional_data[1])
plt.xlabel('Delay ($\mu$s)')
plt.ylabel('Count')
plt.xlim([-5, 5])
plt.savefig('../output/photon16_150kcounts_bidirectional_histogram.eps', bbox_inches = 'tight')
plt.show()

#%%%%


tags = np.load('../data/photon4/70k_countrate_50nsbench/50nswidth_50khz_4vpp2voffset/collected_tags_15032022_16_08_12.npy')
tags_channel_list = np.load('../data/photon4/70k_countrate_50nsbench/50nswidth_50khz_4vpp2voffset/tags_channel_list_15032022_16_08_12.npy')

channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags, tags_channel_list,commonfloor = True)
data=[channel1,channel2]
channel1, channel2 = deli.data_crop(data, 0.2e12)

print('hihi')
bins = 2000
max_delay = 2.5e6
counts, midpoints = cross.cross_corr(channel1, channel2, bins = bins, max_delay = max_delay)
midpoints *= 1e-6

bidirectional_data = np.array([midpoints, counts])
np.savetxt('../output/photon8_70k_data_bidirectional.txt', bidirectional_data)
'''
Plotting Bidirectional Histogram Photon16 70k countrate
'''

bidirectional_data = np.loadtxt('../output/photon8_70k_data_bidirectional.txt')

fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.0f', xaxis_dp = '%.2f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
plt.plot(bidirectional_data[0], bidirectional_data[1])
plt.xlabel('Delay ($\mu$s)')
plt.ylabel('Count')
plt.xlim([-2.5, 2.5])
plt.savefig('../output/photon8_70kcounts_bidirectional_histogram.eps', bbox_inches = 'tight')
plt.show()


#%%%%

tags = np.load('../data/photon8/100k_countrate_50nsbench/50nswidth_50khz_4vpp2voffset/collected_tags_15032022_17_37_57.npy')
tags_channel_list = np.load('../data/photon8/100k_countrate_50nsbench/50nswidth_50khz_4vpp2voffset/tags_channel_list_15032022_17_37_57.npy')

channel1, channel2, channel3, channel4 = mathy.tag_fourchannel_splice(tags, tags_channel_list,commonfloor = True)
data=[channel1,channel2]
channel1, channel2 = deli.data_crop(data, 0.2e12)

print('hihi')
bins = 2000
max_delay = 2.5e6
counts, midpoints = cross.cross_corr(channel1, channel2, bins = bins, max_delay = max_delay)
midpoints *= 1e-6

bidirectional_data = np.array([midpoints, counts])
np.savetxt('../output/photon8_100k_data_bidirectional.txt', bidirectional_data)
'''
Plotting Bidirectional Histogram Photon16 100k countrate
'''

bidirectional_data = np.loadtxt('../output/photon8_100k_data_bidirectional.txt')

fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.0f', xaxis_dp = '%.2f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
plt.plot(bidirectional_data[0], bidirectional_data[1])
plt.xlabel('Delay ($\mu$s)')
plt.ylabel('Count')
plt.xlim([-2.5, 2.5])
plt.savefig('../output/photon8_100kcounts_bidirectional_histogram.eps', bbox_inches = 'tight')
plt.show()

#%%
