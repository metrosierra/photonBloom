#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sympy.functions.combinatorial.numbers import stirling
# from itertools import combinations
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt
from numba import njit
import subroutines.mathematics as mathy
import pandas as pd

'''
D = number of detectors
C = number of observed clicks
N = number of photons in initial state


D = [1,2]
mean = bucket_mean(bucket)
C, N, patches = plt.hist(bucket)
'''
#############################################################

@njit
def PCN(D,C,N):
    combination = mathy.numba_combination(D, C)
    S = mathy.numba_stirling2(C, N)
    if D > 0:
        factorial = mathy.numba_factorial(C) / (D**N)
        P = combination * factorial * S
    elif D == 0:
        P = 0
        
    return P
    

@njit
def binomial(p0,m,n):
    P = mathy.numba_combination(n,m) * (p0**m) * ((1-p0)**(n-m))
    return P

# @njit
def noisy_poisson_pc(mean, noise = 0.001, qe = 0.35, clicks = [0,1,2,3,4,5,6,7,8]):

    maxn = poisson_infN(mean)
    fock_poissonprob = np.exp(-mean)

    computation = fock_poissonprob * noisy_pcn(clicks, photon_number = 0, noise = noise, efficiency = qe) 
    for fockn in range(1, maxn + 1):

        fock_poissonprob *= (mean / fockn)  

        computation += fock_poissonprob * noisy_pcn(clicks, photon_number = fockn, noise = noise, efficiency = qe) 

    return computation

# @njit
def noisy_pcn(clicks, photon_number, noise = 0.01, efficiency = 0.95):
    # noise, efficiency= p
    photon_no = int(np.floor(photon_number))
    # noise = 0.1
    # efficiency = 0.85
    detector_no = len(clicks)
    output = []
    for click in clicks:
        click = int(click)
        sum1 = 0.
        for i in range(click+1):
            p1 = binomial(noise,i,detector_no)
            
            sum2 = 0.
            for j in range(click-i, photon_no+1):
                Pd = (detector_no-i)/detector_no
                p2 = binomial(Pd,j,photon_no)

                sum3 = 0.
                for k in range(click-i, j+1):
                    Pd2 = detector_no-i
                    p3 = binomial(efficiency, k, j) * PCN(Pd2, click-i, k)
                    sum3 += p3
                    
                sum2 += p2 * sum3

            sum1 += p1 * sum2
        output.append(float(sum1))

    return np.array(output)#/np.sum(output)

@njit
def poisson_infN(mean, threshold = 0.95):

    prob = 0
    n = 1
    current_prob = np.exp(-mean)
    prob += current_prob
    satisfied = False
    while not satisfied:

        current_prob *= (mean / n)
        prob += current_prob 
        if prob >= threshold:
            satisfied = True

        else:
            n += 1

    return n

########################################################

D=4
N=np.array([1,2,3,4])
C=3


'''
With Poisson Prior
'''





def P_poissonCN(D,C,N,mean):
    factorial = mathy.numba_factorial(C) / mathy.numba_factorial(N)
    S = mathy.numba_stirling2(C, N)
    if D > 0:
        gamma = D/mean
        denominator = (np.exp(1/gamma)-1)**-C
        P = factorial * S * denominator * (gamma**-N)
    elif D == 0:
        P = 0
        
    # print('P(C|N) = ',P)
    return P


def P_poisson_plot(D,C,N,mean):
    n=np.linspace(0,2)
    P_poisson_plot = P_poissonCN(D,C,N,mean)
    plt.plot(n,P_poisson_plot)
    plt.show()
    

def combination_probability(C,D,N):
    '''
    D = number of possible detection events
    N = total number of possible photons detected at each D
    C = specific number of photons detected
    ''' 
    total = [mathy.numba_combination(D,n) for n in N]

    P = mathy.numba_combination(D,C) / sum(total)
    
#    print('Probability of detecting {c} photons in one detector is {p}'.format(c=C,p=P))
    return P

def joint_probabilities(D,C,N):
    '''
    D = number of possible detection events
    N = total number of possible photons detected at each D
    C = specific number of photons detected
    '''
    
    C=np.array(np.arange(0,C+1))
    N=np.array(np.arange(1,N+1))

#    single_probabilities = [combination_probability(c,D,N) for c in C]
    single_probabilities=[(1/(D+1)) for n in range(len(N)+1)]
    print('Singular probabilities = ',single_probabilities)
    
    photons_i = np.array([0,1,2,3,4])
    photons_j = np.array([0,1,2,3,4])
    
    
    df = pd.DataFrame(photons_i)
    for j in photons_j:
        df[j]=[single_probabilities[j]+single_probabilities[i] for i in photons_i]
    
    print(df)
    
    T = sum(df)
    Tc = sum([df[i].iloc[len(C)-1-i] for i in range(len(C))])
    
    joint_probability = Tc/T
    print(joint_probability)
    
    return joint_probability


def poisson(x,mean,a):
    xfact = sp.special.factorial(x)
    return a * ( (mean**x) / xfact ) * np.exp(-mean)



def bucket_mean(bucket):
    hist = plt.hist(np.concatenate(bucket).ravel(),color='grey')

    x=np.array(list(range(3)))
    y = [i for i in hist[0] if int(i) != 0]

    mean=np.mean(y)
    print('Mean value = {m}'.format(m=mean)) 
    return mean



def poisson_histogram_fit(bucket,savefig=False):

    '''
    Plotting the histogram of the data buckets and fitting with a Poisson distribution.
    '''

    hist = plt.hist(np.concatenate(bucket).ravel(),color='grey')

    x=np.array(list(range(3)))
    y = [i for i in hist[0] if int(i) != 0]

    mean=np.mean(y)
    print('Mean value = {m}'.format(m=mean)) 

    for j, val in enumerate(y):
        p_val = poisson(j,val,1)

        print('Poisson coefficient for bin {n} = {p}.'.format(n=int(j),p=p_val)) 

    plt.hist(np.concatenate(bucket).ravel(),color='grey')
    plt.scatter(x,y,marker='.',color='red',)

    opt, cov = sp.optimize.curve_fit(poisson,x,y,maxfev=10000)

    xval=np.linspace(0,3)
    plt.plot(xval,poisson(xval,*opt),'--',color='red',label='Poisson Fit\n mean={m}\n a={s}'.format(m=int(opt[0]),s=int(opt[1])))
    

    plt.ylabel('Counts')
    plt.xlabel('Number')
    plt.legend()
    plt.ylim(0,5000)

    plt.title('Poisson Fit: 10kcps_50ns_50kHz')

    if savefig is True:
        plt.savefig('output/poissonfit_10kcps_50ns_50kHz.eps')
    plt.show()

    fit_mean = opt[0]
    fit_scale = opt[1]

    print('Poisson Fit Parameters:\n Mean = {m} +/- {me} \n Scale = {a} +/- {ae}'.format(m=opt[0],a=opt[1],me=cov[0,0],ae=cov[1,1]))

    return fit_mean, fit_scale
