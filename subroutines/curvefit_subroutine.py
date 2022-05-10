#!/usr/bin/env python3

import numpy as np
import scipy.odr as odr

import lmfit as limmy

'''
https://docs.scipy.org/doc/scipy/reference/odr.html

Basically, define a function, then chuck it into the fit function along with your x y data

Orthogonal Distance Regression (ODR) is better than scipy optimize because it can accept BOTH x and y errors for fitting

Now experimenting with lmfit as well 
'''



def odrfit(function, x, y, initials, xerr = None, yerr = None, param_mask = np.array([1, 0, 0, 0])):

    # Create a scipy Model object
    model = odr.Model(function)
    # Create a RealData object using our initiated data from above. basically declaring all our variables using the RealData command which the scipy package wants
    inputdata = odr.RealData(x, y, sx = xerr, sy = yerr)
    # Set up ODR with the model and data. ODR is orthogonal distance regression (need to google!)
    odr_setup = odr.ODR(inputdata, model, beta0 = initials, ifixb = param_mask)

    print('\nRunning fit!')
    # Run the regression.
    output = odr_setup.run()
    print('\nFit done!')
    # output.beta contains the fitted parameters (it's a list, so you can sub it back into function as p!)
    print('\nFitted parameters = ', output.beta)
    print('\nError of parameters =', output.sd_beta)

    #now we can calculate chi-square (if you included errors for fitting, if not it's meaningless)


    chisquare = np.sum((y - function(output.beta, x))**2/(function(output.beta, x)))
    chi_reduced = chisquare / (len(x) - len(initials))
    # print('\nReduced Chisquare = ', chi_reduced, 'with ',  len(x) - len(initials), 'Degrees of Freedom')

    return output.beta, output.sd_beta, chi_reduced  


def limmyfit(function, x, y, arg):

    model = limmy.Model(function)

    output = model.fit(y = y, x = x, mean = arg)

    print(output.fit_report())

    
