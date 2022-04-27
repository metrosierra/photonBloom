#!/usr/bin/env python3
#%%
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

import os
from natsort import natsorted

import sys 
sys.path.append('../')
from subroutines import mathematics as mathy
from subroutines import twomultiplex_macroroutine as twosie
from subroutines import retrofit_macroroutine as rexy


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


overall_qe = 0.589###AS OF 26 APRIL 2022 

#%%












#%%