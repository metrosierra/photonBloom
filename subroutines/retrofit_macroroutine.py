#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from subroutines import retrodict_subroutine as probby
from subroutines import curvefit_subroutine as oddy
import subroutines.prettyplot as pretty

def poisson_retrofit(counts, multiplex, filename):

    counts_prob = counts / np.sum(counts)
    clicks = np.arange(0, multiplex+1)
    print(clicks, 'hiiiii')
    mean_guess = np.sum(clicks * np.array(counts_prob)) 
    qe = 0.5

    fig, ax = pretty.prettyplot(figsize = (10, 10), yaxis_dp = '%.2f', xaxis_dp = '%.1f', ylabel = 'Normalised Relative Probability', xlabel = 'Detector Click Count', title = 'Retrodict fit of observed detector click probability, Photon{}, qe{}%'.format(multiplex, qe*100))
    plt.plot(clicks, counts_prob, ls = '--', color='red', label = 'Observed Photon{} Click Distribution'.format(multiplex))

    ### mean, noise, multiplex, qe
    input_param = np.array([mean_guess, 0.0001, multiplex, qe])

    fit_results = oddy.odrfit(probby.noisy_poisson_pc, x = clicks, y = counts_prob, initials = input_param, param_mask = np.array([1, 0, 0, 0]))
    counts_fit = probby.noisy_poisson_pc(fit_results[0], clicks)
    print(counts_fit)
    plt.plot(clicks, counts_fit, ls = '--', label = 'Fit Poissonian light distribution, {:.3f}$\pm${:.3f} mean'.format(fit_results[0][0], fit_results[1][0]))
    ax.legend(fontsize = 15)

    plt.savefig('../output/{}_retrodict_{}percentqe.eps'.format(filename, qe*100))
    plt.savefig('../output/{}_retrodict_{}percentqe.png'.format(filename, qe*100), dpi = 200)
    plt.show(block = False)
    plt.pause(1)
    plt.close()

    return fit_results, counts_fit