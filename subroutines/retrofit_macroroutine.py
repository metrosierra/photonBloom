#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

from subroutines import retrodict_subroutine as probby
from subroutines import curvefit_subroutine as oddy
import subroutines.prettyplot as pretty

def poisson_odr_retrofit(counts, multiplex, filename, qe = 0.5, noise = 0.00001):

    counts_prob = counts / np.sum(counts)
    clicks = np.arange(0, multiplex+1)
    print(clicks, 'hiiiii')
    mean_guess = np.sum(clicks * np.array(counts_prob)) 

    fig, ax = pretty.prettyplot(figsize = (10, 10), yaxis_dp = '%.2f', xaxis_dp = '%.1f', ylabel = 'Normalised Relative Probability', xlabel = 'Detector Click Count', title = 'Retrodict fit of observed probability, Photon{}, qe{}%, noise{:.6f}'.format(multiplex, qe*100, noise))
    plt.plot(clicks, counts_prob, ls = '--', color='red', label = 'Observed Photon{} Click Distribution'.format(multiplex))

    ### mean, noise, multiplex, qe
    input_param = np.array([mean_guess, noise, multiplex, qe])

    fit_results = oddy.odrfit(probby.noisy_poisson_pc_new, x = clicks, y = counts_prob, initials = input_param, param_mask = np.array([1, 0, 0, 0]))
    counts_fit = probby.noisy_poisson_pc_new(fit_results[0], clicks)
    print(counts_fit)
    plt.plot(clicks, counts_fit, ls = '--', label = 'Fit Poissonian light distribution, {:.3f}$\pm${:.3f} mean'.format(fit_results[0][0], fit_results[1][0]))
    ax.legend(fontsize = 15)

    plt.savefig('../output/{}_retrodict_{}percentqe.eps'.format(filename, qe*100))
    plt.savefig('../output/{}_retrodict_{}percentqe.png'.format(filename, qe*100), dpi = 200)
    plt.show(block = False)
    plt.pause(1)
    plt.close()

    return fit_results, counts_fit



def poisson_mle_gradient(counts, multiplex, filename, qe = 0.5, noise = 0.001, threshold = 0.001):

    counts_prob = counts / np.sum(counts)
    clicks = np.arange(0, multiplex+1)
    mean_guess = np.sum(clicks * np.array(counts_prob)) 

    gradient = 1.
    step_size = 2.
    trial = mean_guess
    converged = False

    ref_value = probby.log_mle_pc([trial, noise, multiplex, qe], clicks, counts)

    while not converged:
        trial += gradient * step_size
        next_value = probby.log_mle_pc([trial, noise, multiplex, qe], clicks, counts)

        if next_value <= ref_value:
            gradient *= -1
            step_size /= 3

        if step_size < threshold:
            converged = True
            print('gradient converged!')
        
        ref_value = next_value

    opt_trial = trial

    error_step = 0.1
    gradient = -1.
    left_converged = False
    while not left_converged:
        trial += error_step*gradient
        # print('left trial', trial)
        next_value = probby.log_mle_pc([trial, noise, multiplex, qe], clicks, counts)

        # print(abs(ref_value - next_value), 'hi')

        if ref_value - next_value > 0.5:
            ### remove the step, halve the step for the next run
            trial -= gradient*error_step
            error_step /= 2

        if abs(ref_value - next_value - 0.5) < threshold:
            left_converged = True 
    lefterror = trial - opt_trial

    trial = opt_trial
    error_step = 0.5
    gradient = 1.
    right_converged = False
    while not right_converged:
        trial += error_step*gradient
        next_value = probby.log_mle_pc([trial, noise, multiplex, qe], clicks, counts)

        if ref_value - next_value > 0.5: 
            trial -= gradient*error_step
            error_step /= 2

        if abs(ref_value - next_value - 0.5) < threshold:
            right_converged = True 
    righterror = trial - opt_trial

    opt_fit = probby.noisy_poisson_pc_new([opt_trial, noise, multiplex, qe], clicks)
    ssres = np.sum((opt_fit - counts_prob)**2)
    sstot = np.sum((counts_prob - np.mean(counts_prob))**2)
    rsquare = 1 - ssres/sstot


#########################just plotting###########################
    fig, ax = pretty.prettyplot(figsize = (10, 10), yaxis_dp = '%.2f', xaxis_dp = '%.1f', ylabel = 'Normalised Relative Probability', xlabel = 'Detector Click Count', title = 'Retrodict fit of observed probability, Photon{}, qe{}%, noise{:.6f}'.format(multiplex, qe*100, noise))
    plt.plot(clicks, counts_prob, ls = '--', color='red', label = 'Observed Photon{} Click Distribution'.format(multiplex))

    counts_fit = probby.noisy_poisson_pc_new([opt_trial, noise, multiplex, qe], clicks)
    plt.plot(clicks, counts_fit, ls = '--', label = 'Fit Poissonian, {:.3f}{:.3f}+{:.3f} mean, {:.5f} rsquare'.format(opt_trial, lefterror, righterror, rsquare))
    ax.legend(fontsize = 15)

    plt.savefig('../output/{}_mleretrodict_{}percentqe.eps'.format(filename, qe*100))
    plt.savefig('../output/{}_mleretrodict_{}percentqe.png'.format(filename, qe*100), dpi = 200)
    plt.show(block = False)
    plt.pause(1)
    plt.close()


    return opt_trial, lefterror, righterror, rsquare