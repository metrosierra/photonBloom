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



#%%


overall_qe = 0.589###AS OF 26 APRIL 2022 

#%%



###data
### fig, ax = prettyplot()
### plt.plot 
### plt.save 
### plt.show
### plt.close()




#%%