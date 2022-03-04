#!/usr/bin/env python3

import numpy as np
from scipy.odr import *


'''
https://docs.scipy.org/doc/scipy/reference/odr.html

Basically, define a function, then chuck it into the fit function along with your x y data

Orthogonal Distance Regression (ODR) is better than scipy optimize because it can accept BOTH x and y errors for fitting
'''

def fit(function, x, y, initials, xerr = None, yerr = 1.):

    # Create a scipy Model object
    model = Model(function)
    # Create a RealData object using our initiated data from above. basically declaring all our variables using the RealData command which the scipy package wants
    input = RealData(x, y, sx = xerr, sy = yerr)
    # Set up ODR with the model and data. ODR is orthogonal distance regression (need to google!)
    odr = ODR(input, model, beta0 = initials)

    print('\nRunning fit!')
    # Run the regression.
    output = odr.run()
    print('\nFit done!')
    # output.beta contains the fitted parameters (it's a list, so you can sub it back into function as p!)
    print('\nFitted parameters = ', output.beta)
    print('\nError of parameters =', output.sd_beta)

    #now we can calculate chi-square (if you included errors for fitting, if not it's meaningless)


    # chisquare = np.sum((y - function(output.beta, x))**2/(yerr**2))
    # chi_reduced = chisquare / (len(x) - len(initials))
    # print('\nReduced Chisquare = ', chi_reduced, 'with ',  len(x) - len(initials), 'Degrees of Freedom')



    return output.beta, output.sd_beta, 1.
