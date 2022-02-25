#!/usr/bin/env python3

import sys
import TimeTagger
import socket

import numpy as np
import matplotlib.pyplot as plt
import time

from numba import njit

def gaussian(p, x):
    """
    Returns a scaled Gaussian function for visual comparison of the measured data
    """
    mu, sigma, A = p
    return A/(np.sqrt(2*np.pi)*sigma)*np.exp(-0.5*(x-mu)**2/sigma**2)





@njit
def get_bin_edges(a, bins):
    bin_edges = np.zeros((bins+1,), dtype=np.float64)
    a_min = a.min()
    a_max = a.max()
    delta = (a_max - a_min) / bins
    for i in range(bin_edges.shape[0]):
        bin_edges[i] = a_min + i * delta

    bin_edges[-1] = a_max  # Avoid roundoff error on last point
    return bin_edges


@njit
def compute_bin(x, bin_edges):
    # assuming uniform bins for now
    n = bin_edges.shape[0] - 1
    a_min = bin_edges[0]
    a_max = bin_edges[-1]

    # special case to mirror NumPy behavior for last bin
    if x == a_max:
        return n - 1 # a_max always in last bin

    bin = int(n * (x - a_min) / (a_max - a_min))

    if bin < 0 or bin >= n:
        return None
    else:
        return bin


@njit
def numba_histogram(a, bins):
    hist = np.zeros((bins,), dtype=np.intp)
    bin_edges = get_bin_edges(a, bins)

    for x in a.flat:
        bin = compute_bin(x, bin_edges)
        if bin is not None:
            hist[int(bin)] += 1

    return hist, bin_edges
