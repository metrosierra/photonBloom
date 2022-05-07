#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#%%

import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.cm as cm

import retrodict_subroutine as probby
import curvefit_subroutine as oddy
import retrofit_macroroutine as rexxy
import prettyplot as plot

#%%

'''
Evaluating Fit Values for Photon16 clicks distribution
'''

multiplex = 16
noise = 0.0001
qe = 0.589
#Photon16_150k
photon16_150k = [16, qe, 0.01725, [19., 102., 268., 503., 565., 444., 332., 172., 63., 15., 4., 1., 0., 0., 0., 0., 0.]]

clicks = photon16_150k[3]

clicks_prob = clicks/np.sum(clicks)


x_domain = np.arange(0, multiplex +1)
mean_guess = np.sum(x_domain * np.array(clicks_prob)) 
print(mean_guess)

output = []
trials = np.linspace(np.floor(mean_guess/3),np.floor(mean_guess*3.5), round(mean_guess*3.5))
print(trials)

print('hihihi',probby.log_mle_pc([9.5, noise, multiplex, qe], x_domain, clicks))

fits = rexxy.poisson_mle_gradient(clicks, multiplex, filename = 'pipeline5_test', qe = 0.589, noise = 0.0001, threshold = 0.0001)

print(fits)


#%%
'''
Comparing model for click distribution for different QE
QE in the range 0.1 to 1
'''

qe_range = np.arange(0.1,1,0.1)
error_qefits = []

fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.2f', xaxis_dp = '%.0f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
for l,qe in enumerate(qe_range):
    noise=0.0001
    x_domain = np.arange(0, multiplex )
    fit = probby.noisy_poisson_pc([fits[0], noise, multiplex, qe], x_domain)
    
    error_qefit = 0
    for i,c in enumerate(clicks_prob[:16]):
        if i>0 and c>0:
            err = np.sqrt((c - fit[i])**2)
            error_qefit += err
        else:
            error_qefit +=0
    error_qefits.append(error_qefit/sum(clicks_prob[:16]))
    
    plt.plot(x_domain, fit, label='{}'.format(round(qe,1)), color=cm.Blues(l/10))
x2_domain = np.arange(0, multiplex+1)
plt.plot(x2_domain, clicks_prob, color='black',linestyle='--',label='Observed\nClicks')
plt.legend(title='QE', fontsize=18,title_fontsize=18)
plt.ylabel('Probability')
plt.xlabel('Clicks')
plt.savefig('../output/photon16_150kcounts_varying_qe_comparison.eps')
plt.show()

print('Error between model and fit for varied QE',error_qefits)

'''
Plot of model deviation from true fit for varied QE
'''
fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.2f', xaxis_dp = '%.1f', ylabel = 'Error', xlabel = 'Quantum Efficiency', title = None)
plt.plot(qe_range, error_qefits)
plt.show()


#%%

'''
Comparing model for click distribution for different noise
Noise in the range 0 to 0.25
'''
error_noisefits = []

fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.2f', xaxis_dp = '%.0f', ylabel = 'Counts', xlabel = 'Clicks', title = None)
noise_range = np.arange(0.00001,0.25,0.05)
for i,n in enumerate(noise_range):
    qe=0.589
    x_domain = np.arange(0, multiplex )
    fit = probby.noisy_poisson_pc([fits[0], n, multiplex, qe], x_domain)
    
    error_noisefit = 0
    for j,c in enumerate(clicks_prob[:16]):
        if j>0 and c>0:
            errn = np.sqrt((c - fit[j])**2)
            error_noisefit += errn
        else:
            error_noisefit +=0
    error_noisefits.append(error_noisefit/sum(clicks_prob[:16]))
    
    
    
    plt.plot(x_domain, fit, label='{}'.format(round(n,2)), color=cm.Blues_r(i/5))
x2_domain = np.arange(0, multiplex+1)
plt.plot(x2_domain, clicks_prob, color='black',linestyle='--',label='Observed\nClicks\nQE=0.589')
plt.legend(title='Noise', fontsize=18,title_fontsize=18)
plt.ylabel('Probability')
plt.xlabel('Clicks')
plt.savefig('../output/photon16_150kcounts_varying_noise_comparison.eps')
plt.show()

print('Error between model and fit for varied noise',error_noisefits)

'''
Plot of model deviation from true fit for varied noise
'''

fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.2f', xaxis_dp = '%.3f', ylabel = 'Error', xlabel = 'Noise', title = None)
plt.plot(noise_range, error_noisefits)
plt.show()


#%%

'''
Limitations from Loss

n = iteration of setup starting with n=1 for Photon4, n=2 for Photon8 and so on...
'''

connector_loss = 0.25*5 #dB
delay_lengths = [0, 0.07, 0.21, 0.49, 1.19, 2.87, 6.93, 16.73, 40.39, 97.51]

def splice_loss(n):
    loss_per_unit = 0.001
    splice_loss = (2*n + 2) * loss_per_unit
    return splice_loss

def coupler_loss(n):
    loss_per_unit = 0.1
    coupler_loss = n * loss_per_unit
    return coupler_loss

def fibre_loss(length):
    loss_per_unit = 0.21
    fibre_loss = length * loss_per_unit
    return fibre_loss

def total_loss(loss_per_n):
    reflectance = 1-0.04
    qe = 0.9
    return (net_loss(loss_per_n) * reflectance * qe)


def net_loss(loss_per_n):
    setup_loss = (10**(-loss_per_n/10))
    return setup_loss

N = np.arange(0,8,1)
photon_resolution = 2**(N+1)

loss_per_n = []
net_losses = []
total_losses = []
percentage_losses = []

fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.2f', xaxis_dp = '%.0f', ylabel = 'Loss (dB)', xlabel = 'Photon Number Resolution', title = None)
for n in N:
    combined_loss = splice_loss(n) + coupler_loss(n) + fibre_loss(delay_lengths[n]) + connector_loss
    loss_per_n.append(combined_loss)
    net_losses.append(1-net_loss(combined_loss))
    total_losses.append(total_loss(combined_loss))
    percentage_losses.append(100*(1-net_loss(combined_loss))/(1-total_loss(combined_loss)))

    plt.scatter(2**(n+1), combined_loss,marker='s',color='dodgerblue', label='Combined Loss')
plt.plot(photon_resolution, loss_per_n,linestyle='--',color='dodgerblue')
plt.savefig('../output/photon16_150kcounts_loss_vs_photonresolution.eps')
plt.show()

#%%

'''
Comparing different types of loss with increasing photon number reolution
'''
fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.2f', xaxis_dp = '%.0f', ylabel = 'Loss (dB)', xlabel = 'Photon Number Resolution', title = None)

plt.scatter(photon_resolution, splice_loss(N),marker='s',color='dodgerblue', label='Splice Loss')
plt.plot(photon_resolution, splice_loss(N),linestyle='--',color='dodgerblue')
plt.scatter(photon_resolution, coupler_loss(N),marker='s',color='firebrick', label='Coupler Loss')
plt.plot(photon_resolution, coupler_loss(N),linestyle='--',color='firebrick')
plt.scatter(photon_resolution, fibre_loss(np.array(delay_lengths)[:max(N)+1]),marker='s',color='midnightblue', label='Fibre Loss')
plt.plot(photon_resolution, fibre_loss(np.array(delay_lengths)[:max(N)+1]),linestyle='--',color='midnightblue')

plt.legend(fontsize=20)
plt.savefig('../output/photon16_150kcounts_losstypes.eps')
plt.show()

#%%

'''
Quantum Efficiency with Photon Number Resolution
'''

fig, ax = plot.prettyplot(figsize = (9, 9), yaxis_dp = '%.2f', xaxis_dp = '%.0f', ylabel = 'Quantum Efficiency', xlabel = 'Photon Number Resolution', title = None)
plt.plot(photon_resolution, total_losses,linestyle='--',color='dodgerblue')
plt.scatter(photon_resolution, total_losses, marker = 's', color='dodgerblue', label='PhotonN QE')
plt.legend(fontsize=20)
plt.savefig('../output/photon16_150kcounts_qe_vs_photonresolution.eps')
plt.show()

print(photon_resolution)
print(total_losses)
print(total_losses[3])

print(splice_loss(3) + coupler_loss(3) + fibre_loss(delay_lengths[3]) + connector_loss)
print(splice_loss(3) , coupler_loss(3) , fibre_loss(delay_lengths[3]) , connector_loss)
print(splice_loss(7) , coupler_loss(7) , fibre_loss(delay_lengths[7]) , connector_loss)

'''
Loss becomes dominated by length of fibre optic
where trend line becomes linear
'''

for a,b in zip(photon_resolution, total_losses):
    if b <= 0.5:
        print(a)


#%%


'''
Limitations from Dispersion

--> NEGLIGIBLE
'''


def gv_dispersion(pulse_width, wavelength, fibre_length, dispersion_coefficient):
    c = 299792458*2/3
    
    f1 = 4 * np.log(2) * (wavelength**2) * dispersion_coefficient * fibre_length
    f2 =  (pulse_width**2) * 2 * np.pi * c
    
    disp_width = pulse_width * np.sqrt(1 + ( f1/f2 )**2 )
    
    return disp_width

def max_time_bins(period, disp_width):
    return period / (disp_width)

frequency = 50000 #Hz
period = 1/frequency
pulse_width = 50 * 10**-9 #s
wavelength = 1550 * 10**-9 #m
#fibre_length = 0.48 #km
fibre_length = np.arange(1,1000) #m
dispersion_coefficient = 18 * 10**-9 


def disp_fibre_lengths(fibre_length):
    
    frequency = 50000 #Hz
    period = 1/frequency
    pulse_width = 50 * 10**-9 #s
    wavelength = 1550 * 10**-9 #m
    dispersion_coefficient = 18 * 10**-6
    disp_width = gv_dispersion(pulse_width, wavelength, fibre_length, dispersion_coefficient)

    plt.plot(fibre_length,disp_width)
    plt.xlabel('Fibre Length (km)')
    plt.ylabel('Dispersed Width (ps)')
    plt.title('Pulse width dispersion with fibre length.')
    plt.show()
    
    max_bins = max_time_bins(period, disp_width)
    
    plt.plot(fibre_length,max_bins)
    plt.xlabel('Fibre Length (km)')
    plt.ylabel('Maximum Number of Bins')
    plt.title('Maximum Number of Bins with fibre length.')
    plt.show()
    
frequency = np.arange(0,1000000) #Hz

def disp_pulse_widths(frequency):
    period = 1/frequency
    pulse_widths = 9*10**-12
    fibre_length = period * 299792458*2/3
    wavelength = 1550 * 10**-9 #m
    dispersion_coefficient = 18 * 10**-6
    disp_width = gv_dispersion(pulse_widths, wavelength, fibre_length, dispersion_coefficient)

#    plt.plot(pulse_widths,disp_width)
#    plt.xlabel('Pulse Width (ps)')
#    plt.ylabel('Dispersed Width (ps)')
#    plt.title('Pulse width dispersion with pulse width.')
#    plt.show()
    
    max_bins = max_time_bins(period, disp_width)/ (2*10**5)   #halved to account for FWHM
    
    plt.plot(1/period,max_bins)
    plt.xlabel('Pulse Width (ps)')
    plt.ylabel('Maximum Number of Bins')
    plt.title('Maximum Number of Bins with pulse width.')
    plt.show()
    
    
disp_fibre_lengths(fibre_length)    
disp_pulse_widths(frequency)  

#dif = disp_width - pulse_width
#
#print('Output pulse length = {} ps'.format(disp_width))
#print('Difference = {} ps'.format(dif))



    
    
