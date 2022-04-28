#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import numpy as np
import matplotlib.pyplot as plt
import math

'''
Limitations from Loss

n = iteration of setup starting with n=1 for Photon4, n=2 for Photon8 and so on...
'''

connector_loss = 0.25*5 #dB

def splice_loss(n):
    loss_per_unit = 0.0005
    splice_loss = (2*n + 2) * loss_per_unit
    return splice_loss

def coupler_loss(n):
    loss_per_unit = 0.1
    coupler_loss = n * loss_per_unit
    return coupler_loss

def fibre_loss(n):
    loss_per_unit = 0.21
    fibre_loss = 0.07 * n * loss_per_unit
    return fibre_loss

def total_loss(loss_per_n):
    reflectance = 1-0.04
    qe = 0.9
    return (net_loss(loss_per_n) * reflectance * qe)


def net_loss(loss_per_n):
    setup_loss = (10**(-loss_per_n/10))
    return setup_loss

N = np.arange(1,7,1)

loss_per_n = []
net_losses = []
total_losses = []
percentage_losses = []
qe_losses = []
for n in N:
    combined_loss = splice_loss(n) + coupler_loss(n) + fibre_loss(n) + connector_loss
    loss_per_n.append(combined_loss)
    net_losses.append(1-net_loss(combined_loss))
    total_losses.append(1-total_loss(combined_loss))
    percentage_losses.append(100*(1-net_loss(combined_loss))/(1-total_loss(combined_loss)))
    qe_losses.append(100*(1-(0.9*0.96))/(1-total_loss(combined_loss)))
    
    plt.scatter(n, combined_loss)
    plt.ylabel('Loss (dB)')
    plt.xlabel('n')
plt.show()

photon_resolution = 2**(N+1)

plt.plot(photon_resolution, percentage_losses,linestyle='--',color='black',label='Setup Loss')
#plt.plot(photon_resolution, qe_losses,linestyle='--',color='red',label='Detector Loss')
plt.title('Percentage contribution from setup loss with Photon Number Resolution')
plt.ylabel('Percentage Loss (%)')
plt.xlabel('Photon Number Resolution')
plt.legend()
plt.show()


#%%


'''
Limitations from Dispersion
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



    
    

