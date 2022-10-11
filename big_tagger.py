#!/usr/bin/env python3
"""
Library for client-side control of Swabian Timetagger Ultra

New timetagger clients should inherit the 'BaseTag' class and all key functions 
should be implemented here and not in offspring classes
See the docstrings of the implemented functions for the exisiting scopes.

Todo:
*
*
*
"""

import TimeTagger

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import time

import subroutines.jitter_subroutine as jitty
import subroutines.mathematics as mathy
from subroutines.mathematics import percentsss, starsss

try:
    import data_tools as dt
    _fileloc, _calibrationslib, _adjustmentslib = dt.localfiles()
except ModuleNotFoundError as mex:
    print(f"{mex} -> dev_tools not in python path!")
    _fileloc = _calibrationslib = _adjustmentslib = None


class BaseTag():
    """Underlying TimeTagger class for interfacing with SNSPDs

    Parameters
    ----------
    tagger : TimeTaggerNetwork object, default None
        Timetagger object from Swabian Timetagger package
    savefile : str, default None
        Filename of the file populated with countrate/correlation data
    print_kwargs : bool, default False
        Print list of possible kwargs
    verbose : bool, default False
        Print additional outputs during runtime
    disable_autoconfig : bool, default True
        Disable resetting trigger levels and gate times to values specified in configurations/tagger_config.json
    


    Attributes
    ----------
    fileloc, calibrationslib, adjustmentslib : see data_tools.localfiles()
        Data folder, calibrations folder and adjustments folder locations
    client : Timetagger object
        Timetagger object
    savefile : str
        Current name of measurement save data
    res_mode : dict
        Resolution mode of Timetagger - averages over multipleR TDC channels to reduce jitter
    print_kwargs : bool
        Print list of possible kwargs
    verbose : bool
        Print additional outputs at runtime
    disable_autoconfig : bool
        Disable resetting trigger levels and gate times to values specified in configurations/tagger_config.json
    is_test : bool

    is_on : bool

    corr_running : bool

    allrate_running : bool

    countrate_running : bool

    stream_running : bool

    countrate : np.array()

    data : list

    config : dict

    """

    def __init__(
        self, 
        tagger = None, 
        savefile = None,
        print_kwargs = False,
        verbose = False,
        disable_autoconfig = True,
        **kwargs):

        fileloc, calibrationslib, adjustmentslib = (
        _fileloc,
        _calibrationslib,
        _adjustmentslib,
        )
        self.client = tagger
        self.res_modes = {'Standard': TimeTagger.Resolution.Standard, 'HighResA': TimeTagger.Resolution.HighResA,
                     'HighResB': TimeTagger.Resolution.HighResB, 'HighResC': TimeTagger.Resolution.HighResC}
        self.savefile = savefile
        self.print_kwargs = print_kwargs
        self.verbose = verbose
        self.disable_autoconfig = disable_autoconfig

        self.is_test = False
        self.is_on = True

        self.corr_running = False
        self.allrate_runnning = False 
        self.countrate_running = False
        self.stream_running = False
        self.countrate = np.array([[0, 0], [0, 0]])
        self.data = []
        self.config = {}


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
    
    def set_manualconfig(self, channels):
        """Manualy set selected channels to the same trigger level [V], deadtime [ps], event divider [int] and LED power [1/0] """

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
            self.set_trigger(channel = channel, level = trigger)
            self.config['channel{}'.format(channel)]['trigger'] = trigger

            self.set_deadtime(channel = channel, deadtime = deadtime)
            self.config['channel{}'.format(channel)]['deadtime'] = deadtime

            self.set_eventdivider(channel = channel, divider = divider)
            self.config['channel{}'.format(channel)]['divider'] = divider

            self.set_led(turnon = turnon)
            self.config['ledstate'] = turnon

        print('Channels {} configured! Check out the current configuration below:'.format(channels))
        # print(json.dumps(self.config, indent = 4))


    def set_autoconfig(self):
        """Automatically set trigger levels, deadtimes, event dividers and ledstate from configurations/tagger_config.json file"""

        with open(os.path.join(os.path.dirname(__file__),'configurations','tagger_config.json')) as jsondata:

            self.config = json.load(jsondata)
            print(config)
            for i in range(1, 5):
                trigger = self.config['channel{}'.format(i)]['trigger']
                deadtime = self.config['channel{}'.format(i)]['deadtime']
                divider = self.config['channel{}'.format(i)]['divider']
                turnon = self.config['ledstate']

                self.set_trigger(channel = i, level = trigger)
                self.set_deadtime(channel = i, deadtime = deadtime)
                self.set_eventdivider(channel = i, divider = divider)
                self.set_led(turnon = turnon)

        return self.config
    

######################   hardware configuration methods   ################################

    ####about 0.08V for ref
    def set_trigger(self, channel, level):
        self.client.setTriggerLevel(channel = channel, voltage = level)
        if self.verbose:
            print('\n Trigger level set at {}V for channels {}'.format(level, channel))
        return self

    ####about 100ns for ref
    def set_deadtime(self, channel, deadtime):
        self.client.setDeadtime(channel = channel, deadtime = deadtime)
        if self.verbose:
            print('\n Deadtime set at {}ps for channels {}'.format(deadtime, channel))
        return self

    def set_eventdivider(self, channel, divider):
        self.client.setEventDivider(channel = channel, divider = divider)
        if self.verbose:
            print('\n Event divider set at {} events for channels {}'.format(divider, channel))
        return self

    def set_testsignal(self, channels):
        if not self.is_test:
            self.tagger.setTestSignal(channels, True)
            if self.verbose:
                print('Test mode set! Activating test signals on channels {}'.format(channels))
            self.is_test = True
        else:
            self.tagger.setTestSignal(channels, False)
            if self.verbose:
                print('Test mode unset! deactivating test signals on channels {}'.format(channels))
            self.is_test = False

    ###### it's christmas!!!!!!!!!!!
    def set_led(self, turnon = True):

        self.is_on = turnon
        if self.is_on:
            self.client.setLED(1)
        else:
            self.client.setLED(0)
        return self

    def set_ledtoggle(self):

        if self.is_on:
            self.client.setLED(0)
            self.is_on = False
        else:
            self.client.setLED(1)
            self.is_on = True
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

        if type(channels) is int:
            channels = [channels]
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
                if self.verbose:
                    print(f'Measured count of channel(s) {channels} in counts:')
                    print(self.countrate)

        return self.countrate


    ###subclassing the CountRate measurement class
    def get_countrate(self, startfor = int(1e12), channels = [1, 2, 3, 4]):

        if type(channels) is int:
            channels = [channels]
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

                if self.verbose:
                    print(f'Measured total total count rate of channel {channels}  in counts/s:')
                    print(self.allrate)

        return self.allrate

    ### full auto and cross correlation
    def get_correlation(self, startfor = int(1E12), channels = [1, 2], binwidth = 1000, n = 1000):

        if type(channels) is int:
            channels = [channels]
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

    ### 2D correlation vs time-delay from a trigger channel
    def triggered_correlation(self, 
                              trigger_channel = 2 , 
                              corr_channel1 = 3, 
                              corr_channel2 = 4, 
                              binwidth_ns=500, 
                              n_values=20,
                              runtime = int(1e12)):
        # Create Correlation measurements and use SynchronizedMeasurements to start them easily
        sm = TimeTagger.SynchronizedMeasurements(self.client)
        
        if (n_values % 2) == 0:
            n_values+=1
        
        midpoint = n_values/2+0.5
        
        # Make virtual channels that are delayed by each multiple of the bin widths
        # then take coincidences for corr_channel1(2) and the trigger_channel
        # this creates virtual channels that can then be correlated with corr_channel2(1) 
        ydelayed = []
        ycoincidences = []
        y2coincidences = []
        
        channel1delayed = TimeTagger.DelayedChannel(sm.getTagger(), 
                                                  input_channel=corr_channel1, 
                                                  delay=midpoint*binwidth_ns*1000)
        channel2delayed = TimeTagger.DelayedChannel(sm.getTagger(), 
                                                  input_channel=corr_channel2, 
                                                  delay=midpoint*binwidth_ns*1000)
        
        for i in range(n_values):
            ydelayed.append(TimeTagger.DelayedChannel(sm.getTagger(), 
                                                      input_channel=trigger_channel, 
                                                      delay=(i)*binwidth_ns*1000))
            ycoincidences.append(TimeTagger.Coincidence(sm.getTagger(), [channel1delayed.getChannel(),ydelayed[i].getChannel()],
                                             coincidenceWindow = binwidth_ns*1000,
                                             timestamp = TimeTagger.CoincidenceTimestamp.ListedFirst))
            y2coincidences.append(TimeTagger.Coincidence(sm.getTagger(), [channel2delayed.getChannel(),ydelayed[i].getChannel()],
                                             coincidenceWindow = binwidth_ns*1000,
                                             timestamp = TimeTagger.CoincidenceTimestamp.ListedFirst))
        
        # Measure correlations between the delayed triggered virtual channels and corr_channel2(1)
        corr_list = []
        corr2_list = []
        
        for i in range(n_values):
            corr_list.append(TimeTagger.Correlation(sm.getTagger(),
                                                    ycoincidences[i].getChannel(),
                                                    channel2delayed.getChannel(), 
                                                    binwidth = binwidth_ns*1000,
                                                    n_bins = n_values))
            corr2_list.append(TimeTagger.Correlation(sm.getTagger(),
                                                    y2coincidences[i].getChannel(),
                                                    channel1delayed.getChannel(), 
                                                    binwidth = binwidth_ns*1000,
                                                    n_bins = n_values))
        
        # Run for runtime
        sm.startFor(runtime, clear=True)
        sm.waitUntilFinished()
        
        outputdata = np.zeros(shape=(n_values,n_values))
        
        for i in range(n_values):
            with corr_list[i] as corr, corr2_list[i] as corr2:
                dat = np.add(corr.getData(), corr2.getData())
                #print(np.array(dat))
                outputdata[i]=np.array(dat)                
        return outputdata
     
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
