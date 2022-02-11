#!/usr/bin/env python3

import sys
import TimeTagger
import socket

import numpy as np
import matplotlib.pyplot as plt
import time


def gaussian(p, x):
    """
    Returns a scaled Gaussian function for visual comparison of the measured data
    """
    mu, sigma, A = p

    return A/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*(x-mu)**2/sigma**2)
