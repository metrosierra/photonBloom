#!/usr/bin/env python3

import matplotlib.pyplot as plt
from matplotlib.ticker import FormatStrFormatter


def prettyplot(figsize = (9, 6), yaxis_dp = '%.1f', xaxis_dp = '%.2f', ylabel = 'Y Axis', xlabel = 'X Axis', title = 'Title'):


    fig, ax = plt.subplots(figsize = figsize)
    # params = {
    #     'text.usetex': True,
    #     'font.family': 'lmodern',
    # }
    # plt.rcParams.update(params)
    # plt.rcParams['text.latex.preamble'] = [r'\usepackage{lmodern}']

    ax.tick_params(labelsize = 20, direction = 'in', length = 6, width = 1, bottom = True, top = True, left = True, right = True)
    ax.yaxis.set_major_formatter(FormatStrFormatter(yaxis_dp))
    ax.xaxis.set_major_formatter(FormatStrFormatter(xaxis_dp))

    ax.set_xlabel(xlabel, fontsize = 20)
    ax.set_ylabel(ylabel, fontsize = 20)
    ax.set_title(title, fontsize = 20)

    return fig, ax