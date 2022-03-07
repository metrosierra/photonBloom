#!/usr/bin/env python3

###%%%%%%%%%%%%%%%%%%%%%%
import TimeTagger as tt
import numpy as np
from datetime import datetime
import time
import json
import threading

import matplotlib.pyplot as plt

from client_object import TagClient
from liveplotter import Plumeria

# from live_plot import bloomPlot
from subroutines.mathematics import percentsss, starsss

@percentsss
@starsss
def bienvenue():
    print('This is the datalogger script for the Photon Spot Nanowire detector, where the Blossom class instantiates client objects and plotting objects to achieve datalogging UX')

bienvenue()

dummy_config = {"channel1": {"index": 1}, "channel2": {"index": 2}, "channel3": {"index":3}, "channel4": {"index": 4}, "ledstate": 0}

###loosely datalogger/plotter class
class Lotus():

    def __init__(self):

        print('Initiliasing prototype usage file powered by TagClient methods')
        self.plotting = False
        self.plot_freeze = False
        self.config = dummy_config
        self.create_networktagger('192.168.0.2')



    def create_networktagger(self, ip_address):
        self.spot0 = TagClient(ip_address)
        return self.spot0 

    def __enter__(self):
        return self

    @percentsss
    @starsss
    def __exit__(self, exception_type, exception_value, exception_traceback):
        print('Ciao bella ciao bella ciao ciao')
        del self.spot0
  
####################### Hardware config macros #######################

    def set_manualconfig(self, channels):

        trigger = float(input('\nPlease input the trigger level in volts!!\n'))
        print('{}V Received!'.format(trigger))
        deadtime = float(input('\nPlease input the deadtimes in picoseconds!\n'))
        print('{}ps Received!'.format(deadtime))
        divider =  round(int(input('\nPlease input the divider integer!\n')))
        print('{} divider order received!'.format(divider))
        turnon = round(int(input('\nLED Power: 1/0????\n')))
        print('Logic of {} received!'.format(turnon))


        for channel in channels:

            channel = round(channel)
            self.spot0.set_trigger(channel = channel, level = trigger)
            self.config['channel{}'.format(channel)]['trigger'] = trigger

            self.spot0.set_deadtime(channel = channel, deadtime = deadtime)
            self.config['channel{}'.format(channel)]['deadtime'] = deadtime

            self.spot0.set_eventdivider(channel = channel, divider = divider)
            self.config['channel{}'.format(channel)]['divider'] = divider

            self.spot0.set_led(turnon = turnon)
            self.config['ledstate'] = turnon


        print('Channels {} configured! Check out the current configuration below:'.format(channels))
        # print(json.dumps(self.config, indent = 4))


    def set_autoconfig(self):

        with open('configuration/tagger_config.json') as jsondata:

            self.config = json.load(jsondata)
            print(self.config)
            for i in range(1, 5):

                trigger = self.config['channel{}'.format(i)]['trigger']
                deadtime = self.config['channel{}'.format(i)]['deadtime']
                divider = self.config['channel{}'.format(i)]['divider']
                turnon = self.config['ledstate']

                self.spot0.set_trigger(channel = i, level = trigger)
                self.spot0.set_deadtime(channel = i, deadtime = deadtime)
                self.spot0.set_eventdivider(channel = i, divider = divider)
                self.spot0.set_led(turnon = turnon)

        return self

    def stop_plot(self):
        self.plotting = False


####################### Measurement + Plotting Macros #######################

    def start_countplot_protocol(self):

        self.tag_counter(startfor = -1, channels = [1, 2], binwidth = 1e11, n = 100, save = False)

        threading.Thread(target = self.create_liveplot, args = (self.spot0.countrate,), daemon = True).start()


    def create_liveplot(self, targetdata, xlabel = 'X Axis', ylabel = 'Y Axis', title = 'Unknown Plot', refresh_interval = 0.0001, initial_xydata = [[0.], [0.]]):

        self.plotting = True
        with Plumeria(title = title, refresh_interval = refresh_interval, initial_xydata = initial_xydata) as plume:

            plume.set_data(initial_xydata)
            plume.set_xlabel(xlabel)
            plume.set_ylabel(ylabel)
            
            while self.plotting:
                    plume.set_data(targetdata)
                    plume.update()

        print('Plotting session killed!')


    def tag_counter(self, startfor, channels, binwidth = 1000, n = 1000, save = False):

        if startfor == -1:
            print('Persisting Counter class measurement!!!')
            threading.Thread(target = self.spot0.get_count, args = (startfor, channels, binwidth, n, save, ), daemon = True).start()

        elif startfor > 0.:
            counts = self.spot0.get_count(startfor, channels, binwidth, n)

            if save:
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H_%M_%S")
                np.save('output/lastframe_bincounts_width{}ps_n{}_{}'.format(binwidth, n, dt_string), counts)

            return counts



    def tag_correlation(self, startfor, channels, binwidth = 1000, n = 1000, save = False):

        if startfor == -1:
            print('Persisting Counter class measurement!!!')
            threading.Thread(target = self.spot0.get_correlation, args = (startfor, channels, binwidth, n,), daemon = True).start()
            return self.spot0.corr_counts, self.spot0.corr_running

        elif startfor > 0.:
            hist_data = self.spot0.get_correlation(startfor, channels, binwidth, n)

            if save:
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H_%M_%S")
                np.save('output/correlated_width{}ps_n{}_{}'.format(binwidth, n, dt_string), hist_data)

            return hist_data


    def streamdata(self, startfor, channels, buffer_size = 100000, update_rate = 0.0001, verbose = True):
        tags, channel_list = self.spot0.streamdata(startfor = startfor, channels = channels, buffer_size = buffer_size, update_rate = update_rate, verbose = verbose)

        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H_%M_%S")
        np.save('output/collected_tags_{}'.format(dt_string), tags)
        np.save('output/tags_channel_list_{}'.format(dt_string), channel_list)

        return tags, channel_list




if __name__ == '__main__':
    with Lotus() as lot:
        print("\n\n################## With Lotus as lot ###################\n\n")
        import code; code.interact(local=locals())
###%%%%%%%%%%%%%%%%%%%%%%

#
#
