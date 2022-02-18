#!/usr/bin/env python3


###%%%%%%%%%%%%%%%%%%%%%%
import TimeTagger as tt
import numpy as np
from datetime import datetime
import json

import matplotlib.pyplot as plt

from client_object import TagClient
# from live_plot import bloomPlot



###loosely datalogger/plotter class
class Blossom():

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

    def set_manualconfig(self, channels):

        trigger = float(input('\nPlease input the trigger level in volts!!\n'))
        print('{}V Received!'.format(trigger))
        deadtime = float(input('\nPlease input the deadtimes in picoseconds!\n'))
        print('{}ps Received!'.format(deadtime))
        divider =  round(input('\nPlease input the divider integer!\n'))
        print('{} divider order received!')
        turnon = round(input('\nLED Power: 1/0????\n'))
        print('Logic of {} received!'.format(turnon))


        for channel in channels:

            channel = round(channel)
            self.spot0.set_trigger(channel = channel, level = trigger)
            self.config['channel{}'.format(channel)]['trigger'] = trigger

            self.spot0.set_deadtime(channel = channel, deadtime = deadtime)
            self.config['channel{}'.format(channel)]['deadtime'] = deadtime

            self.spot0.set_eventdivider(channel = channel, divider = divider)
            self.config['channel{}'.format(channel)]['divider'] = divider

            self.spot0.set_led(channel = channel, turnon = turnon)
            self.config['ledstate'] = turnon


        print('Channels {} configured! Check out the current configuration below:'.format(channels))
        print(json.dumps(self.config, indent = 4))





    def set_autoconfig(self):

        with open('tagger_config.json') as jsondata:

            self.config = json.load(jsondata)

            for i in range(1,5):

                self.spot0.set_trigger(channel = i, level = self.config['channel{}'.format(i)]['trigger'])
                self.spot0.set_deadtime(channel = i, level = self.config['channel{}'.format(i)]['trigger'])




            self.config['channel{}'.format(channel)]['deadtime'] = deadtime

            self.spot0.set_eventdivider(channel = channel, divider = divider)
            self.config['channel{}'.format(channel)]['divider'] = divider

            self.spot0.set_led(channel = channel, turnon = turnon)















    def streamdata(self, startfor, channels, buffer_size = 100000, update_rate = 0.0001, verbose = True):
        tags, channel_list = self.spot0.streamdata(startfor = startfor, channels = channels, buffer_size = buffer_size, update_rate = update_rate, verbose = verbose)

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


if __name__ == '__main__':
    with Blossom() as bloss:
        print("\n\n################## With Blossom as bloss ###################\n\n")
        import code; code.interact(local=locals())
###%%%%%%%%%%%%%%%%%%%%%%

#
#
