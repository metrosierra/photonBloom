#!/usr/bin/env python3

"""
Library for data viewing over LAN of Swabian Time Tagger Ultra

Methods use big_tagger class BaseTag and subclass TagClient to expose TimeTagger methods over LAN

Todo:
*
*
*
"""

import numpy as np
import os
from datetime import datetime
import json
import threading
import time
import sys
from PyQt5.QtWidgets import QApplication
from client_object import TagClient
from liveplotter import Plumeria
from liveplotter_mark2 import RoseApp

from subroutines.mathematics import percentsss, starsss

@percentsss
@starsss
def bienvenue():
    print('This is the datalogger script for the Photon Spot Nanowire detector, where the Lotus class instantiates client objects and plotting objects to achieve datalogging UX')

bienvenue()

dummy_config = {"channel1": {"index": 1}, "channel2": {"index": 2}, "channel3": {"index":3}, "channel4": {"index": 4}, "ledstate": 0}

###loosely datalogger/plotter class
class Lotus():

    def __init__(self, **kwargs):

        print('Initiliasing prototype usage file powered by TagClient methods')
        self.config = dummy_config
        self.create_networktagger('192.168.0.2', **kwargs)

        if self.spot0.disable_autoconfig == False:
            self.spot0.set_autoconfig()
            print('Automatically configuring Time Tagger based on JSON config file...change using set_manualconfig method')

        self.app = QApplication(sys.argv)
        self.rose0 = RoseApp()


    def create_networktagger(self, ip_address, **kwargs):
        self.spot0 = TagClient(ip_address, **kwargs)
        return self.spot0 

    def __enter__(self):
        return self

    # def gui_thread(self):
    #     self.app = QApplication(sys.argv)
    #     self.rose0 = RoseApp()
    #     self.app.exec_()
    #     print('Rose GUI Main Window Activated! Add plots at will!')

    @percentsss
    @starsss
    def ciao(self):
        print('Ciao')
        print('Datalogger Object Destroyed')

    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.spot0.__exit__(exception_type, exception_value, exception_traceback)
        self.ciao()

            
####################### Measurement + Plotting Macros #######################
### now we assume that if we startfor = -1 we open a live plot
### we generate a new boolean flag in a list that tracks the "identity"
### of each measurement and thus plotting session (for each function call).
### Closing the liveplot toggles the specific flag in the list
### and thus also causes the measurement thread to end as well
####################################################################################
### Measurement + Data Saving Protocols

    def binwidth_study(self):

        self.tag_correlation(startfor=600e12, channels = [3,4], binwidth_ns =0.1, n=100000, save=True)
        self.tag_correlation(startfor=300e12, channels = [3,4], binwidth_ns =0.5, n=50000, save=True)
        self.tag_correlation(startfor=300e12, channels = [3,4], binwidth_ns =1, n=25000, save=True)
        self.tag_correlation(startfor=300e12, channels = [3,4], binwidth_ns = 5, n=5000, save=True)
        self.tag_correlation(startfor=60e12, channels = [3,4], binwidth_ns = 10, n=2400, save=True)
        self.tag_correlation(startfor=60e12, channels = [3,4], binwidth_ns = 20, n=1200, save=True)
        self.tag_correlation(startfor=60e12, channels = [3,4], binwidth_ns = 50, n=600, save=True)
        self.tag_correlation(startfor=60e12, channels = [3,4], binwidth_ns = 100, n=300, save=True)

        print('100ns mark reached!!!')

        self.tag_correlation(startfor=30e12, channels = [3,4], binwidth_ns = 125, n=300, save=True)
        self.tag_correlation(startfor=30e12, channels = [3,4], binwidth_ns = 150, n=300, save=True)
        self.tag_correlation(startfor=30e12, channels = [3,4], binwidth_ns = 175, n=300, save=True)
        self.tag_correlation(startfor=30e12, channels = [3,4], binwidth_ns = 200, n=300, save=True)
        self.tag_correlation(startfor=30e12, channels = [3,4], binwidth_ns = 400, n=300, save=True)

        print('400ns mark reached!!!')

        self.tag_correlation(startfor=30e12, channels = [3,4], binwidth_ns = 800, n=300, save=True)
        self.tag_correlation(startfor=30e12, channels = [3,4], binwidth_ns = 1000, n=300, save=True)
        self.tag_correlation(startfor=30e12, channels = [3,4], binwidth_ns = 5000, n=300, save=True)
        self.tag_correlation(startfor=30e12, channels = [3,4], binwidth_ns = 10000, n=300, save=True)



    def run_pet(self, startfor=60e12):

        self.tag_correlation(startfor=startfor, channels = [3,4], binwidth_ns =10, n=4800, save=True)
        self.tag_triggered_correlation(startfor=startfor, channels = [1,3,4], binwidth_ns = 50, n=3000, stacks = 40, save=True)

        # self.tag_correlation(startfor=startfor, channels = [1,3], binwidth_ns =10, n=2400, save=True)
        # self.tag_correlation(startfor=startfor, channels = [1,4], binwidth_ns =10, n=2400, save=True)

    def run_trig(self, startfor=30e12):

        self.tag_triggered_correlation(startfor=startfor, channels = [1,3,4], binwidth_ns =100, n=300, stacks = 20, save=True)
        self.tag_triggered_correlation(startfor=startfor, channels = [1,3,4], binwidth_ns =150, n=200, stacks = 20, save=True)
        self.tag_triggered_correlation(startfor=startfor, channels = [1,3,4], binwidth_ns =200, n=100, stacks = 20, save=True)


    def tag_counter(self, startfor, channels, binwidth_ns = 1e9, n = 10, save = True):
        
        if startfor == -1:
            print('Persisting Counter measurement class! Close live plot to exit this!!')

            self.spot0.count_running.append(True)
            identity = len(self.spot0.count_running) - 1
            threading.Thread(target = self.spot0.get_count, args = (startfor, channels, binwidth_ns, n, identity), daemon = True).start()

            qthread_args = {
                'data_func': self.spot0.return_count, 
                'data_kill_func': self.spot0.switchoff_count,
                'identity': identity,
                'plot_no': len(channels)
            }

            self.rose0.new_window(refresh_interval = 0.1, 
                                    title = 'Rolling Count Rate Plot', 
                                    xlabel = 'Time (s)', 
                                    ylabel = 'Counts/s', 
                                    **qthread_args)

            return 

        elif type(startfor) is int or type(startfor) is float:
            counts = self.spot0.get_count(startfor, channels, binwidth_ns, n)

            if save:
                if not os.path.exists('output/'): os.makedir('output/')
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H_%M_%S")
                np.save('output/lastframe_bincounts_width{}ps_n{}_{:.1e}time_{}'.format(binwidth_ns, n, startfor, dt_string), counts)
            return counts

        elif type(startfor) is list and len(startfor) == 1 and type(startfor[0]) is int or type(startfor[0]) is float:
            segment_ps = 30e12
            cycles = int(np.round(startfor[0]/segment_ps))
            print('Running segmented data run! {} cycles of 30s each'.format(cycles))
            print('Starting cycle 1 of {}'.format(cycles))
            counts = self.spot0.get_count(segment_ps, channels, binwidth_ns, n)
            if save:
                if not os.path.exists('output/'): os.makedir('output/')
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H_%M_%S")
                filename = 'output/lastframe_bincounts_width{}ps_n{}_{:.1e}time_{}'.format(binwidth_ns, n, startfor, dt_string)
                np.save(filename, counts)
            print('{}ps elapsed!'.format(segment_ps))

            for i in range(cycles-1):
                print('Starting cycle {} of {}'.format(i+2, cycles))
                counts += self.spot0.get_count(segment_ps, channels, binwidth_ns, n)
                if save:
                    np.save(filename, counts)
                print('{}ps elapsed!'.format((i+2)*segment_ps))

            return counts

        else: 
            print('Invalid startfor argument! Must be -1, int or float, or list of int or float of length 1')



    def tag_correlation(self, startfor, channels, binwidth_ns = 1000, n = 100, save = True):

        if startfor == -1:
            print('Persisting XCorrelation measurement class! Close live plot to exit this!!')
            self.spot0.corr_running.append(True)
            identity = len(self.spot0.corr_running) - 1
            threading.Thread(target = self.spot0.get_correlation, args = (startfor, channels, binwidth_ns, n, identity), daemon = True).start()
            
            qthread_args = {
                'data_func': self.spot0.return_corr_counts, 
                'data_kill_func': self.spot0.switchoff_corr_counts,
                'identity': identity,
                'plot_no': 1
            }

            self.rose0.new_window(refresh_interval = 0.1, 
                                    title = 'Cross Correlation Plot', 
                                    xlabel = 'Delay', 
                                    ylabel = 'Counts', 
                                    **qthread_args)

            return

        elif type(startfor) is int or type(startfor) is float:
            corr = self.spot0.get_correlation(startfor, channels, binwidth_ns, n)

            if save:
                if not os.path.exists('output/'): os.makedir('output/')
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H_%M_%S")
                np.save('output/correlated_width{}ns_n{}_ch{}_{:.1e}time_{}'.format(binwidth_ns, n, channels, startfor, dt_string), corr)
            return corr

        elif type(startfor) is list and len(startfor) == 1 and type(startfor[0]) is int or type(startfor[0]) is float:
            segment_ps = 30e12
            cycles = int(np.round(startfor[0]/segment_ps))
            print('Running segmented data run! {} cycles of 30s each'.format(cycles))
            print('Starting cycle 1 of {}'.format(cycles))
            corr = self.spot0.get_correlation(segment_ps, channels, binwidth_ns, n)
            if save:
                if not os.path.exists('output/'): os.makedir('output/')
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H_%M_%S")
                filename = 'output/correlated_width{}ns_n{}_ch{}_{:.1e}time_{}'.format(binwidth_ns, n, channels, startfor, dt_string)
                np.save(filename, corr)
            print('{}ps elapsed!'.format(segment_ps))

            for i in range(cycles-1):
                print('Starting cycle {} of {}'.format(i+2, cycles))
                corr += self.spot0.get_correlation(segment_ps, channels, binwidth_ns, n)
                if save:
                    np.save(filename, corr)
                print('{}ps elapsed!'.format((i+2)*segment_ps))

            return corr

        else: 
            print('Invalid startfor argument! Must be -1, int or float, or list of int or float of length 1')

    def tag_triggered_correlation(self, startfor, channels, binwidth_ns = 100, n = 100, stacks = 20, save = True):

        if startfor == -1:
            print('Persisting TrigXCorrelation measurement class! Close live plot to exit this!!')
            self.spot0.trig_corr_running.append(True)
            identity = len(self.spot0.trig_corr_running) - 1
            threading.Thread(target = self.spot0.get_triggered_correlation, args = (startfor, channels, binwidth_ns, n, stacks, identity), daemon = True).start()

            qthread_args = {
                'data_func': self.spot0.return_trig_corr_counts, 
                'data_kill_func': self.spot0.switchoff_trig_corr_counts,
                'identity': identity,
                'plot_no': stacks
            }

            self.rose0.new_window(refresh_interval = 0.1, 
                                    title = 'Triggered Cross Correlation Plot', 
                                    xlabel = 'Delay', 
                                    ylabel = 'Counts', 
                                    **qthread_args)

            return

        elif type(startfor) is int or type(startfor) is float:
            trigcorr = self.spot0.get_triggered_correlation(startfor, channels, binwidth_ns, n, stacks)

            if save:
                if not os.path.exists('output/'): os.makedir('output/')
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H_%M_%S")
                np.save('output/trigcorrelated_width{}ns_n{}_ch{}_{:.1e}time_{}'.format(binwidth_ns, n, channels, startfor, dt_string), trigcorr)
            return trigcorr

        elif type(startfor) is list and len(startfor) == 1 and type(startfor[0]) is int or type(startfor[0]) is float:
            segment_ps = 30e12
            cycles = int(np.round(startfor[0]/segment_ps))
            print('Running segmented data run! {} cycles of {}ps each'.format(cycles, segment_ps))
            print('Starting cycle 1 of {}'.format(cycles))
            trigcorr = self.spot0.get_triggered_correlation(segment_ps, channels, binwidth_ns, n, stacks)
            if save:
                if not os.path.exists('output/'): os.makedir('output/')
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H_%M_%S")
                filename = 'output/trigcorrelated_width{}ns_n{}_ch{}_{:.1e}time_{}'.format(binwidth_ns, n, channels, startfor, dt_string)
                np.save(filename, trigcorr)
            print('{}ps elapsed!'.format(segment_ps))

            for i in range(cycles-1):
                print('Starting cycle {} of {}'.format(i+2, cycles))
                trigcorr += self.spot0.get_triggered_correlation(segment_ps, channels, binwidth_ns, n, stacks)
                if save:
                    np.save(filename, trigcorr)
                print('{}ps elapsed!'.format((i+2)*segment_ps))

            return trigcorr
        
        else: 
            print('Invalid startfor argument! Must be -1, int or float, or list of int or float of length 1')



    def tag_sweep_correlation(self, startfor = -1, channels = [2,3,4], binwidth_ns = 2, n = 6000, step_no = 5, gatewindow_ns = 75e6, save = True):

        if startfor == -1:
            print('Persisting SweepXCorrelation measurement class! Close live plot to exit this!!')
            self.spot0.sweep_corr_running.append(True)
            identity = len(self.spot0.sweep_corr_running) - 1
            threading.Thread(target = self.spot0.get_sweep_correlation, args = (startfor, channels, binwidth_ns, n, step_no, gatewindow_ns, identity), daemon = True).start()

            qthread_args = {
                'data_func': self.spot0.return_sweep_corr_counts, 
                'data_kill_func': self.spot0.switchoff_sweep_corr_counts,
                'identity': identity,
                'plot_no': step_no
            }

            self.rose0.new_multiwindow(refresh_interval = 0.1, 
                                    title = 'Swept Cross Correlation Plot', 
                                    xlabel = 'Delay', 
                                    ylabel = 'Counts', 
                                    **qthread_args)

            return

        elif type(startfor) is int or type(startfor) is float:
            sweepcorr = self.spot0.get_sweep_correlation(startfor, channels, binwidth_ns, n, step_no, gatewindow_ns)

            if save:
                if not os.path.exists('output/'): os.makedir('output/')
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H_%M_%S")
                np.save('output/sweepcorrelated_width{}ns_n{}_gate{}ns_ch{}_{:.1e}time_{}'.format(binwidth_ns, n, gatewindow_ns, channels, startfor, dt_string), sweepcorr)
            return sweepcorr

        elif type(startfor) is list and len(startfor) == 1 and type(startfor[0]) is int or type(startfor[0]) is float:
            segment_ps = 30e12
            cycles = int(np.round(startfor[0]/segment_ps))
            print('Running segmented data run! {} cycles of 30s each'.format(cycles))
            print('Starting cycle 1 of {}'.format(cycles))
            sweepcorr = self.spot0.get_sweep_correlation(segment_ps, channels, binwidth_ns, n, step_no, gatewindow_ns)

            if save:
                if not os.path.exists('output/'): os.makedir('output/')
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H_%M_%S")
                filename = 'output/sweepcorrelated_width{}ns_n{}_gate{}ns_ch{}_{:.1e}time_{}'.format(binwidth_ns, n, gatewindow_ns, channels, startfor, dt_string)
                np.save(filename, sweepcorr)
            print('{}ps elapsed!'.format(segment_ps))

            for i in range(cycles-1):
                print('Starting cycle {} of {}'.format(i+2, cycles))
                sweepcorr += self.spot0.get_sweep_correlation(segment_ps, channels, binwidth_ns, n, step_no, gatewindow_ns)
                if save:
                    np.save(filename, sweepcorr)
                print('{}ps elapsed!'.format((i+2)*segment_ps))

            return sweepcorr
        
        else: 
            print('Invalid startfor argument! Must be -1, int or float, or list of int or float of length 1')



    def tag_streamdata(self, startfor, channels, buffer_size = 100000, update_rate = 0.0001, verbose = True):
        tags, channel_list = self.spot0.streamdata(startfor = startfor, channels = channels, buffer_size = buffer_size, update_rate = update_rate, verbose = verbose)

        if not os.path.exists('output/'): os.mkdir('output/')
        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H_%M_%S")
        np.save('output/collected_tags_{}'.format(dt_string), tags)
        np.save('output/tags_channel_list_{}'.format(dt_string), channel_list)

        return tags, channel_list


####################################################################################

if __name__ == '__main__':
    with Lotus() as lot:
        print("\n\n################## With Lotus as lot ###################\n\n")
        import code; code.interact(local=locals())
