#!/usr/bin/env python3
import numpy as np
import matplotlib.pyplot as plt

import subroutines.delay_tracking as deli
import subroutines.multiplex_macroroutine as molly
import subroutines.prettyplot as pretty

def twomultiplex(bidata, chopsize, chop_no, binwidth, peno, multiplex, filename, delayno = 350, pulseno = 50):

    channel1fixed, channel2fixed = deli.data_crop(bidata, chopsize * (chop_no + 2))
    channel1chops = deli.channel_chop(channel1fixed, chopsize)
    channel2chops = deli.channel_chop(channel2fixed, chopsize)

    signo = int(multiplex/2-1) * (delayno + pulseno) - delayno
    print('Signal bucket width for photon{}'.format(multiplex), signo)

    output1 = molly.sig_chops_multiplex(channel1chops, chop_no = chop_no, binwidth = binwidth, sig_bin_no = signo, sig_threshold = 1, period_no = peno, multiplex = multiplex/2)
    output2 = molly.sig_chops_multiplex(channel2chops, chop_no = chop_no, binwidth = binwidth, sig_bin_no = signo, sig_threshold = 1, period_no = peno, multiplex = multiplex/2)

    ceiling = min([len(output1), len(output2)])
    aggregate = output1[:ceiling] + output2[:ceiling]


    fig, ax = pretty.prettyplot(figsize = (10, 10), yaxis_dp = '%.0f', xaxis_dp = '%.0f', ylabel = 'Counts', xlabel = 'Detector Click Count', title = 'Observed Photon{} Click Distribution'.format(multiplex))
    output3 = plt.hist(aggregate, density = False, bins = np.arange(-0.5, multiplex + 1.5))
    final_counts = output3[0]
    print(final_counts)
    plt.title('{}'.format(final_counts))

    plt.legend()

    plt.savefig('../output/{}.eps'.format(filename))
    plt.savefig('../output/{}.png'.format(filename), dpi = 200)
    plt.show(block = False)
    plt.pause(1)
    plt.close()

    return final_counts