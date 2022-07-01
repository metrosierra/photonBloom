#!/usr/bin/env python3
import numpy as np
import os
from datetime import datetime
import json
import threading

from client_object import TagClient
from liveplotter import Plumeria
from liveplotter_mark2 import RoseApp, PetalWindow

from subroutines.mathematics import percentsss, starsss

@percentsss
@starsss
def bienvenue():
    print('This is the datalogger script for the Photon Spot Nanowire detector, where the Lotus class instantiates client objects and plotting objects to achieve datalogging UX')

bienvenue()

dummy_config = {"channel1": {"index": 1}, "channel2": {"index": 2}, "channel3": {"index":3}, "channel4": {"index": 4}, "ledstate": 0}

###loosely datalogger/plotter class
class Lotus():

    def __init__(self):

        print('Initiliasing prototype usage file powered by TagClient methods')
        self.config = dummy_config
        self.create_networktagger('192.168.0.2')

        self.set_autoconfig()
        print('Automatically configuring Time Tagger based on JSON config file...change using set_manualconfig method')


    def create_networktagger(self, ip_address):
        self.spot0 = TagClient(ip_address)
        return self.spot0 

    def __enter__(self):
        return self

    @percentsss
    @starsss
    def ciao(self):
        print('Ciao')
        print('Datalogger Object Destroyed')



    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.spot0.__exit__(exception_type, exception_value, exception_traceback)
        self.ciao()
###################### # Hardware config macros #######################

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

        with open(os.path.join(os.path.dirname(__file__),'configurations','tagger_config.json')) as jsondata:

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

    def stop_plot(self, target = 'counter'):
        if target == 'counter':
            self.spot0.countrate_running = False 

        elif target == 'correlation':
            self.spot0.corr_running = False 
            

####################### Measurement + Plotting Macros #######################

### Plotting protocols
####################################################################################

    
    
    
    
    
    
    
    def start_countplot_protocol(self, channels = [1, 2], binwidth = 1e12, n = 20):
        
        # Make binwidth addressable outside of start_countplot_protocol TODO! unify class variables
        
        self.binwidth = binwidth
        
        if self.spot0.countrate_running:
            print('Countrate plot already opened! Please kill it first before another instance!')
        
        else:
            self.tag_counter(startfor = -1, channels = channels, binwidth = binwidth, n = n, save = False)
            threading.Thread(target = self.countplot, args = ('Time (s)', 'Counts', 'Live Countrate Plot', 0.1, len(channels)), daemon = True).start()

        return None

    def countplot(self, xlabel = 'X Axis', ylabel = 'Y Axis', title = 'Unknown Plot', refresh_interval = 0.1, plot_no = 4):

        with Plumeria(title = title, xlabel = xlabel, ylabel = ylabel, refresh_interval = refresh_interval, plot_no = plot_no) as plume:

            plume.set_xlabel(xlabel)
            plume.set_ylabel(ylabel)

            while self.spot0.countrate_running:
                xaxis = np.arange(len(self.spot0.countrate[0]))
                
                for q in range(plot_no):
                    plume.set_data([xaxis, self.spot0.countrate[q] / self.binwidth * 1e12], q)
                
                plume.update()

        print('Countrate Plotting Instance Killed!')
        return None
####################################################################################

    def start_corrplot_protocol(self, channels = [1, 2], binwidth = 10e3, n = 100):
        
        if self.spot0.corr_running:
            print('Correlation plot already opened! Please kill it first before another instance!')
        
        else:
            self.tag_correlation(startfor = -1, channels = channels, binwidth = binwidth, n = n, save = False)
            threading.Thread(target = self.corrplot, args = ('Time (s)', 'Counts', 'Live Countrate Plot'), daemon = True).start()
        
        return None 
 
    def corrplot(self, xlabel = 'X Axis', ylabel = 'Y Axis', title = 'Unknown Plot', refresh_interval = 0.1):

        plot_no = 1
        with Plumeria(title = title, xlabel = xlabel, ylabel = ylabel, refresh_interval = refresh_interval, plot_no = plot_no) as plume:

            plume.set_xlabel(xlabel)
            plume.set_ylabel(ylabel)
  
            while self.spot0.corr_running:
                xaxis = np.arange(len(self.spot0.corr_counts))
                for q in range(plot_no):
                    plume.set_data([xaxis, self.spot0.corr_counts], q)
                
                plume.update()

        print('Correlation Plotting Instance Killed!')
        return None
####################################################################################


### Measurement + Data Saving Protocols

    def tag_counter(self, startfor, channels, binwidth = 1000, n = 1000, save = False):
        

        if startfor == -1:
            print('Persisting Counter class measurement!!!')
            threading.Thread(target = self.spot0.get_count, args = (startfor, channels, binwidth, n), daemon = True).start()

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

        elif startfor > 0.:
            corr = self.spot0.get_correlation(startfor, channels, binwidth, n)

            if save:
                now = datetime.now()
                dt_string = now.strftime("%d%m%Y_%H_%M_%S")
                np.save('output/correlated_width{}ps_n{}_{}'.format(binwidth, n, dt_string), corr)

            return corr

    def tag_streamdata(self, startfor, channels, buffer_size = 100000, update_rate = 0.0001, verbose = True):
        tags, channel_list = self.spot0.streamdata(startfor = startfor, channels = channels, buffer_size = buffer_size, update_rate = update_rate, verbose = verbose)

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
