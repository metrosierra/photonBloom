#!/usr/bin/env python3


###%%%%%%%%%%%%%%%%%%%%%%
import TimeTagger as tt
import numpy as np

import matplotlib.pyplot as plt

from client_object import TagClient
# from live_plot import bloomPlot



###loosely datalogger/plotter class
class spotty():


    def __init__(self):

        print('Initiliasing prototype usage file powered by TagClient methods')
        self.plotting = False
        self.plot_freeze = False

        self.spot0 = TagClient('192.168.0.2')

        input('please type in start_test into the server terminal session and press ENTER')
        print('attempting to stream data')
        data = self.spot0.streamdata(startfor = int(2E7), channels = [1, 2], buffer_size = 1000000, update_rate = 0.0001, verbose = False)


    def start_plot_protocol(self, refresh_time, seconds):

        threading.Thread(target = self.live_plot, args = (refresh_time,), daemon = True).start()
        threading.Thread(target = self.log_timer, args = (seconds,), daemon = False).start()

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
