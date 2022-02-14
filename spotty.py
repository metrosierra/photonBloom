#!/usr/bin/env python3


###%%%%%%%%%%%%%%%%%%%%%%
import TimeTagger as tt
import numpy as np
from datetime import datetime

import matplotlib.pyplot as plt

from client_object import TagClient
# from live_plot import bloomPlot



###loosely datalogger/plotter class
class spotty():


    def __init__(self):

        print('Initiliasing prototype usage file powered by TagClient methods')
        self.plotting = False
        self.plot_freeze = False
        self.config = {}
        self.spot0 = TagClient('192.168.0.2')

        input('please type in start_test into the server terminal session and press ENTER')


    def start_plot_protocol(self, refresh_time, seconds):

        threading.Thread(target = self.live_plot, args = (refresh_time,), daemon = True).start()
        threading.Thread(target = self.log_timer, args = (seconds,), daemon = False).start()

    def start_config_checklist(self, channel):

        print('\n\nWarning: this is a global config function = will overwrite all\n\n')
        self.config['Active Channels'] = channel
        trigger = input('\nPlease input the trigger level in volts!!\n')
        self.spot0.setTriggerLevel(channel = channel, voltage = trigger)
        self.config['Common Trigger'] = trigger

        deadtime = input('\nPlease input the deadtimes in picoseconds!\n')
        deadtime = self.config['Common Deadtime']
        self.spot0.setTriggerLevel(channel = channel, deadtime = deadtime)

        divider =  input('\nPlease input the divider integer!\n')
        self.config['Common Eventdivider'] = divider
        self.client.setEventDivider(channel = channel, divider = divider)

        bitmask = input('\nLED logic bitmask\n')
        self.config['LED Logic Levels'] = bitmask
        self.client.setLED(channel = channel, bitmask = bitmask)

    def streamdata(self, startfor, channel, buffer_size = 100000, update_rate = 0.0001, verbose = True):
        tags, channel_list = self.spot0.streamdata(startfor = startfor, channels = channel, buffer_size = buffer_size, update_rate = update_rate, verbose = verbose)

        now = datetime.now()
        dt_string = now.strftime("%d%m%Y_%H_%M_%S")
        np.save('output/collected_tags_{}'.format(dt_string), tags)
        np.save('output/tags_channel_list_{}'.format(dt_string), channel_list)

        return tags, channel_list

    def log_timer(self, seconds):
        time.sleep(seconds)
        print("Time's up!!")
        self.log_pause()

    def log_pause(self):
        self.plot_freeze = True

    def log_continue(self):
        self.plot_freeze = False

    def live_plot(self, refresh_time):

        self.frame = 0

        self.plotting = True
        self.save = True

        #x, y = something

        ylabel = 'Signal Counts'
        xlabel = 'Time Trace (s)'
        title = 'Timetagger Count Acquisition'
        refresh_time = refresh_time

        with bloomPlot(title, refresh_time) as bp:

            bp.set_xlabel(xlabel)
            bp.set_ylabel(ylabel)
            bp.x = x
            bp.y = y
            start = time.time()
            while self.plotting:
                if not self.plot_freeze:
                    x, y = target_func(self.save)
                    bp.y = y
                try:
                    bp.update()
                except:
                    print('Data error, buffeting')

                self.frame += 1



    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass

hi = spotty()
###%%%%%%%%%%%%%%%%%%%%%%

#
#
