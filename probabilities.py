#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from sympy.functions.combinatorial.numbers import stirling, bell
from itertools import combinations
import math
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt

'''
D = number of detectors
C = number of observed clicks
N = number of photons in initial state

'''

D = [1,2]
mean = bucket_mean(bucket)
C, N, patches = plt.hist(bucket)


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

    opt, cov = curve_fit(poisson,x,y,maxfev=10000)

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


def PCN(D,C,N):
    combination_list = list(combinations(D,C))
    S = [stirling(C, i,kind=2,signed=False) for i in range(N)]
    factorial = int(math.factorial(C) / (len(D)**N))
        
    P = [factorial*combination_list[0][i]*S[i] for i in range(len(D))]
    
    print('P(C|N) = ',P)
    return P
    

'''
With Poisson Prior
'''

def P_poissonCN(D,C,N,mean):
    factorial = int( math.factorial(C) / math.factorial(N))
    S = [stirling(C, i,kind=2,signed=False) for i in range(N)]
    denominator = [int((np.exp(mean/d)-1)**-C) for d in D]
    
    P = [factorial * S[i] * int((D[i]/mean)**-N) * denominator for i in range(len(D))]
    
    print('P_poisson(C|N) = ',P)
    return P


def P_poisson_plot(D,C,N,mean):
    n=np.linspace(0,2)
    P_poisson_plot = P_poissonCN(D,C,N,mean)
    plt.plot(n,P_poisson_plot)
    plt.show()

