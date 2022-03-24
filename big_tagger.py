#!/usr/bin/env python3

import TimeTagger

import numpy as np
import matplotlib.pyplot as plt
import time

import subroutines.jitter_subroutine as jitty
import subroutines.mathematics as mathy
from subroutines.mathematics import percentsss, starsss

class BaseTag():
    def __init__(self, tagger):

        self.client = tagger
        self.res_modes = {'Standard': TimeTagger.Resolution.Standard, 'HighResA': TimeTagger.Resolution.HighResA,
                     'HighResB': TimeTagger.Resolution.HighResB, 'HighResC': TimeTagger.Resolution.HighResC}

        self.istest = False
        self.eyesopen = True

        self.corr_running = False
        self.allrate_runnning = False 
        self.countrate_running = False
        self.stream_running = False
        self.countrate = np.array([[0, 0], [0, 0]])

    @percentsss 
    @starsss
    def welcome_message(self):
        print('Hello tout le monde!')

    def __enter__(self):
        return self

    @percentsss 
    @starsss
    def __exit__(self, exception_type, exception_value, exception_traceback):
        TimeTagger.freeTimeTagger(self.client)
        print('Sarabada, tomo yo-')
        print('Tagger Client Object Destroyed')

########################### initialisation/info grabbing methods #################################


    def get_info(self):

        self.model = self.client.getModel()

        if self.model not in ['Time Tagger 20', 'Time Tagger Ultra']:
            raise ValueError('Device currently not supported')

        self.license_info = self.client.getLicenseInfo().split()
        self.edition = self.license_info[self.license_info.index('Edition:')+1]
        if self.edition == 'High-Res':
            self.modes = self.res_modes.keys()
        else:
            self.modes = [self.model + ' ' + self.edition]  # Only single mode available  # Only single mode available
        print('Connected to a ' + self.model + ' ' + self.edition)
        print('Serial: {}'.format(self.client.getSerial()))

        return self

    def get_methods_naive(self):
        self.method_list = [attribute for attribute in dir(self) if callable(getattr(self, attribute)) and attribute.startswith('__') is False]

        return self.method_list


    def get_jitter(self):

        jitty.warmup(self.client)
        measured_jitters = []
        within_specs = []
        measured_channels = []

        # We loop over the different HighRes modes, if available. Or only use the single available mode.
        for i, mode in enumerate(self.modes):
            if self.edition == 'High-Res':
                print('Setting the Time Tagger into {} mode'.format(mode))
                # We need to first free the Time Tagger to initiate it again with a different HighRes modes
                TimeTagger.freeTimeTagger(self.client)
                self.client = TimeTagger.createTimeTaggerNetwork(address = self.target_ip, resolution = self.res_modes[mode])
            print('Single channel RMS jitter is specified with {} ps'.format(jitty.jitter_specs_rms[mode]))
            if 'HighRes' in mode:
                channels_available = self.client.getChannelList(TimeTagger.ChannelEdge.HighResRising)
            else:
                channels_available = self.client.getChannelList(TimeTagger.ChannelEdge.Rising)
            print('The available channel numbers are {}'.format(channels_available))
            self.client.setTestSignal(channels_available, True)  # Reactivating the test signals
            print('Measuring for 30 seconds')

            # Retrieving the measured data
            indices, data, meas_chan = jitty.synchronized_correlation_measurement(self.client, channels_available, duration = int(30e12))
            measured_channels.append(meas_chan)
            measured_jitters.append(np.full(len(meas_chan), '', dtype=object))
            within_specs.append(np.full(len(meas_chan), '', dtype=object))
            print('Measurement complete.\nNow evaluating the data')
            time.sleep(1)

            fig, ax = plt.subplots()  # Create a plot to visualize the TWO channel jitter
            # Looping over the measurement data, evaluating the single channel RMS jitter from it and displaying the results
            # For the visual comparison a Gaussian with standard deviation = sqrt(2)*specified_RMS_jitter is used
            # since we look at the two-channel jitter
            for j, (ind, dat) in enumerate(zip(indices, data)):
                std, mean = jitty.get_jitter_and_delay(ind, dat)
                measured_jitters[i][j] = std
                within_specs[i][j] = std < jitty.jitter_specs_rms[mode]
                print('channel numbers ' + measured_channels[i][j] +
                      ': measured single channel RMS jitter: {} ps, within specifications: {}'.format(std, within_specs[i][j]))
                if j == len(data)-1:  # make only label for last curve to not clutter the plot.
                    label = 'measured jitter'
                else:
                    label = None
                ax.plot(ind-mean, dat/1e3, label = label)
            ax.set_xlim((-jitty.jitter_specs_rms[mode]*5, jitty.jitter_specs_rms[mode]*5))
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('kCounts')

            ax.plot(ind, mathy.gaussian([0, jitty.jitter_specs_rms[mode]*np.sqrt(2), np.sum(data)/len(data)/1e3], ind),
                    color = 'k', ls = '--', label = 'specified jitter')

            ax.set_title('Visual comparison of the measured two-channel jitters to specifications')
            ax.legend(loc = 1)
            plt.show()
            print('\n')

        print('Plotting a summary of the results')

        # Summary of the measured RMS jitters for a single channel in a table
        colLabels = ['single channel\nRMS jitter (ps)', 'within\nspecifications']
        fig, axes = plt.subplots(1, len(self.modes), figsize=(len(self.modes)*7, len(measured_channels[0])/1.6+0.7))
        axes = np.atleast_1d(axes)
        for i in np.arange(len(self.modes)):
            content = np.hstack((measured_jitters[i].reshape(-1, 1), within_specs[i].reshape(-1, 1)))
            colors = np.full_like(content, 'white')
            colors[within_specs[i] == True, 1] = 'C2'
            colors[within_specs[i] == False, 1] = 'C3'
            the_table = axes[i].table(cellText = content, colLabels = colLabels, rowLabels = measured_channels[i],
                                      loc = 'center', cellColours = colors, cellLoc = 'center')
            the_table.scale(0.78, 2.2)
            axes[i].axis('tight')
            axes[i].axis('off')
            if self.edition == 'High-Res':
                axes[i].set_title(list(self.res_modes.keys())[i], size = 12, pad = 7)
            else:
                axes[i].set_title(self.model + ' ' + self.edition, size = 12, pad = 7)
        axes[0].text(-0.05, 0.5, 'channel combination', rotation = 90, transform = axes[0].transAxes, va = 'center', size = 12)
        if self.edition == 'High-Res':
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        plt.show()

        return measured_jitters, within_specs, measured_channels

######################   hardware configuration methods   ################################

    ####about 0.08V for ref
    def set_trigger(self, channel, level):
        self.client.setTriggerLevel(channel = channel, voltage = level)
        print('\n Trigger level set at {}V for channels {}'.format(level, channel))
        return self

    ####about 100ns for ref
    def set_deadtime(self, channel, deadtime):
        self.client.setDeadtime(channel = channel, deadtime = deadtime)
        print('\n Deadtime set at {}ps for channels {}'.format(deadtime, channel))
        return self

    def set_eventdivider(self, channel, divider):
        self.client.setEventDivider(channel = channel, divider = divider)
        print('\n Event divider set at {} events for channels {}'.format(divider, channel))
        return self

    def set_testsignal(self, channels):
        if not self.istest:
            self.tagger.setTestSignal(channels, True)
            print('Test mode set! Activating test signals on channels {}'.format(channels))
            self.istest = True
        else:
            self.tagger.setTestSignal(channels, False)
            print('Test mode unset! deactivating test signals on channels {}'.format(channels))
            self.istest = False

    ###### it's christmas!!!!!!!!!!!
    def set_led(self, turnon = True):

        self.eyesopen = turnon
        if self.eyesopen:
            self.client.setLED(1)
        else:
            self.client.setLED(0)
        return self

    def set_ledtoggle(self):

        if self.eyesopen:
            self.client.setLED(0)
            self.eyesopen = False
        else:
            self.client.setLED(1)
            self.eyesopen = True
        return self

######################   measurement methods   ################################


    def overflow_check(self):
        # Check for overflows
        self.overflows = self.client.getOverflows()
        if self.overflows == 0:
            print("All incoming data has been processed.")
        else:
            print("""{} data blocks are lost.\nBlock loss can happen during the USB transfer from the Time Tagger
        to the Time Tagger Server and/or during the data transfer over the network from the Time Tagger Server to the client.
        Overflows are caused by exceeding the processing power (CPU) on the client and/or the server,
        the USB bandwidth, or the network bandwidth.""".format(self.overflows))


    def get_count(self, startfor = int(1e12), channels = [1, 2], binwidth = 1000, n = 1000):

        # With the TimeTaggerNetwork object, we can set up a measurement as usual
        with TimeTagger.Counter(self.client, channels, binwidth, n) as compte:

            if startfor == -1 and self.countrate_running == False:
                self.countrate_running = True
                compte.start()
                
                while self.countrate_running:
                    self.countrate = compte.getData(rolling = True)
                    # print(self.countrate)
                compte.stop()

            elif startfor == -1 and self.countrate_running == True:
                print('Counter object instance already exists!!! Please destroy it first')
                self.countrate = np.array([0])


            elif startfor > 0.:

                compte.startFor(startfor)
                compte.waitUntilFinished()
                self.countrate = compte.getData()
                print('Measured count of channel 1-4 in counts:')

        return self.countrate


    ###subclassing the CountRate measurement class
    def get_countrate(self, startfor = int(1e12), channels = [1, 2, 3, 4]):

        # With the TimeTaggerNetwork object, we can set up a measurement as usual
        with TimeTagger.Countrate(self.client, channels) as cr:

            if startfor == -1 and self.allrate_running == False:
                self.allrate_running = True
                cr.start()
                
                while self.allrate_running:
                    self.allrate = cr.getData()
                cr.stop()

            elif startfor == -1 and self.allrate_running == True:
                print('Countrate object instance already exists!!! Please destroy it first')
                self.allrate = 0.

            elif startfor > 0.:

                cr.startFor(startfor)
                cr.waitUntilFinished()
                self.allrate = cr.getData()

                print('Measured total total count rate of channel 1-4 in counts/s:')
                print(self.allrate)

        return self.allrate

    ### full auto and cross correlation
    def get_correlation(self, startfor = int(1E12), channels = [1, 2], binwidth = 1000, n = 1000):

        with TimeTagger.Correlation(self.client, channels[0], channels[1], binwidth, n) as corr:
            
            if startfor == -1 and self.corr_running == False:
                self.corr_running = True
                corr.start()
                while self.corr_running:
                    self.corr_counts = corr.getData()
                corr.stop()

            elif startfor == -1 and self.corr_running == True:
                print('Correlation object instance already exists!!! Please destroy it first')
                self.corr_counts = np.array([0.])

            elif startfor > 0.:
                corr.startFor(startfor)
                corr.waitUntilFinished()
                self.corr_counts = corr.getData()
                print(self.corr_counts)

        ### 1d np array (int)
        return self.corr_counts

#TODO!
    def filewrite(self, startfor = int(5E11), channels = [1, 2, 3, 4]):
        pass
        synchronized = TimeTagger.SynchronizedMeasurements(self.client)

        # This FileWriter will not start automatically, it waits for 'synchronized'
        filewriter = TimeTagger.FileWriter(synchronized.getTagger(), "filewriter", channels)


    ###subclassing the timetagstream class which is under measurement classes
    ###buffer size is memory buffer allocated that is read and destroyed with each
    ###getData() call.
    ###the rate of getData() call must be quick relative to buffer size to avoid
    ###data overflow
    ###startfor is in picoseconds
    def streamdata(self, startfor = int(5E11), channels = [1, 2, 3, 4], buffer_size = 1000000, update_rate = 0.0001, verbose = True):

        format_string = '{:>8} | {:>17} | {:>7} | {:>14} | {:>13}'
        print(format_string.format('TAG #', 'EVENT TYPE', 'CHANNEL', 'TIMESTAMP (ps)', 'MISSED EVENTS'))
        print('---------+-------------------+---------+----------------+--------------')
        event_name = ['0 (TimeTag)', '1 (Error)', '2 (OverflowBegin)', '3 (OverflowEnd)', '4 (MissedEvents)']


        with TimeTagger.TimeTagStream(tagger = self.client, n_max_events = buffer_size, channels = channels) as stream:

            if startfor == -1:
                pass
                # stream.start()
            
            elif startfor > 0.:
                stream.startFor(startfor)

            event_counter = 0
            chunk_counter = 1

            collected_tags = np.array([1.])
            tags_channel_list = np.array([1.])

            while stream.isRunning():
                # getData() does not return timestamps, but an instance of TimeTagStreamBuffer
                # that contains more information than just the timestamp
                data = stream.getData()
                print(data.getTimestamps())
                # print(len(data.getTimestamps()))
                time.sleep(update_rate)
                # print(data.getEventTypes())
                if data.size:
                    # With the following methods, we can retrieve a numpy array for the particular information:
                    channel = data.getChannels()            # The channel numbers
                    timestamps = data.getTimestamps()       # The timestamps in ps
                    overflow_types = data.getEventTypes()   # TimeTag = 0, Error = 1, OverflowBegin = 2, OverflowEnd = 3, MissedEvents = 4
                    missed_events = data.getMissedEvents()  # The numbers of missed events in case of overflow

                    collected_tags = np.concatenate((collected_tags, timestamps))
                    tags_channel_list = np.concatenate((tags_channel_list, channel))

                    if verbose:
                        print(format_string.format(*" "*5))
                        heading = ' Start of data chunk {} with {} events '.format(chunk_counter, data.size)
                        extra_width = 69 - len(heading)
                        print('{} {} {}'.format("="*(extra_width//2), heading, "="*(extra_width - extra_width//2)))
                        print(format_string.format(*" "*5))

                        print(format_string.format(event_counter + 1, event_name[overflow_types[0]], channel[0], timestamps[0], missed_events[0]))
                        if data.size > 1:
                            print(format_string.format(event_counter + 1, event_name[overflow_types[1]], channel[1], timestamps[1], missed_events[1]))
                        if data.size > 3:
                            print(format_string.format(*["..."]*5))
                        if data.size > 2:
                            print(format_string.format(event_counter + data.size, event_name[overflow_types[-1]], channel[-1], timestamps[-1], missed_events[-1]))

                    event_counter += data.size
                    chunk_counter += 1

        print(missed_events, 'events missed!!!')
        return collected_tags[1:], tags_channel_list[1:]
