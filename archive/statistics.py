#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar  8 11:44:41 2022

@author: sabrinaperrenoud
"""
import numpy as np
import scipy.stats as stats
import itertools

def probability(x, mu, n, qe,pd):
    '''
    x = number of clicks
    mu = mean photon number
    n = detector elements
    qe = quantum efficiency of detector
    pd = darak count probability per time bin
    
    '''
    
    P = itertools.combinations(n,x) * np.exp(-mu*qe) * ((1-pd)**n) * ( ((np.exp(mu*qe/n))/(1-pd)) -1 )**x
    
    return P


def mean_mle(x, n, qe,pd):
    '''
    x = expectation value of data
    n = deetctor elements
    qe = quantum efficiency of detector
    pd = darak count probability per time bin
    
    '''
    
    mean = (-n/qe) * np.ln( (n-x) / (n-(pd*n)))
    
    return mean