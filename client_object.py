#!/usr/bin/env python3

import sys
import TimeTagger
import socket

import numpy as np
import matplotlib.pyplot as plt
import time

import jitter_subroutine as jitty
import mathematics as mathy

class TagClient():


    def __init__(self, ip_address = '192.168.0.2'):

        self.res_modes = {'Standard': TimeTagger.Resolution.Standard, 'HighResA': TimeTagger.Resolution.HighResA,
                     'HighResB': TimeTagger.Resolution.HighResB, 'HighResC': TimeTagger.Resolution.HighResC}

        self.target_ip = ip_address
        print("Search for Time Taggers on the network...")
        # Use the scanTimeTaggerServers() function to search for Time Tagger servers in the local network
        self.serverlist = TimeTagger.scanTimeTaggerServers()

        print("{} servers found.".format(len(self.serverlist)))
        print(self.serverlist)

        self.server_handshake()
        print('\nTimetagger object initialising...assuming it is reading from PhotonSpot nanowire single-photon detector...will prompt about detector gain settings in a bit\n')
        self.get_methods_naive()
        print('Here are the available class methods to be used in interactive mode')

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        TimeTagger.freeTimeTagger(self.client)
        print('Ciao!')

########################### initialisation/info grabbing methods #################################

    def server_handshake(self):
        try:
            self.target_server_info = TimeTagger.getTimeTaggerServerInfo('{}'.format(self.target_ip))
            print('Information about Time Tagger server on {}:'.format(self.target_ip))
            print(self.target_server_info)

        except RuntimeError:
            raise Exception('No Time Tagger server available on {} and the default port 41101.'.format(self.target_ip))

        self.client = TimeTagger.createTimeTaggerNetwork(self.target_ip)
        print('Connecting to the server on localhost.')
        print('Server handshake successful!!!!!!!!!!!')
        self.get_info(self.client)


    def get_info(self, taggerself):

        self.model = taggerself.getModel()

        if self.model not in ['Time Tagger 20', 'Time Tagger Ultra']:
            raise ValueError('Device currently not supported')

        self.license_info = taggerself.getLicenseInfo().split()
        self.edition = self.license_info[self.license_info.index('Edition:')+1]
        if self.edition == 'High-Res':
            self.modes = self.res_modes.keys()
        else:
            self.modes = [self.model + ' ' + self.edition]  # Only single mode available  # Only single mode available
        print('Connected to a ' + self.model + ' ' + self.edition)
        print('Serial: {}'.format(taggerself.getSerial()))

        return self


    def get_methods_naive(self):
        self.method_list = [attribute for attribute in dir(self) if callable(getattr(self, attribute)) and attribute.startswith('__') is False]

        return self.method_list

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

    def get_jitter(self):

        jitty.warmup(self.client)
        measured_jitters = []
        within_specs = []
        measured_channels = []

        # We loop over the different HighRes modes, if available. Or only use the single available mode.
        for i, mode in enumerate(self.modes):
            if edition == 'High-Res':
                print('Setting the Time Tagger into {} mode'.format(mode))
                # We need to first free the Time Tagger to initiate it again with a different HighRes modes
                TimeTagger.freeTimeTagger(tagger)
                self.client = TimeTagger.createTimeTaggerNetwork(address = self.target_ip, resolution = self.res_modes[mode])
            print('Single channel RMS jitter is specified with {} ps'.format(jitter_specs_rms[mode]))
            if 'HighRes' in mode:
                channels_available = self.client.getChannelList(TimeTagger.ChannelEdge.HighResRising)
            else:
                channels_available = self.client.getChannelList(TimeTagger.ChannelEdge.Rising)
            print('The available channel numbers are {}'.format(channels_available))
            self.client.setTestSignal(channels_available, True)  # Reactivating the test signals
            print('Measuring for 30 seconds')
            # Retrieving the measured data
            indices, data, meas_chan = synchronized_correlation_measurement(self.client, channels_available, duration=int(30e12))
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
                within_specs[i][j] = std < jitter_specs_rms[mode]
                print('channel numbers ' + measured_channels[i][j] +
                      ': measured single channel RMS jitter: {} ps, within specifications: {}'.format(std, within_specs[i][j]))
                if j == len(data)-1:  # make only label for last curve to not clutter the plot.
                    label = 'measured jitter'
                else:
                    label = None
                ax.plot(ind-mean, dat/1e3, label = label)
            ax.set_xlim((-jitter_specs_rms[mode]*5, jitter_specs_rms[mode]*5))
            ax.set_xlabel('Time (ps)')
            ax.set_ylabel('kCounts')

            ax.plot(ind, mathy.gaussian([0, jitter_specs_rms[mode]*np.sqrt(2), np.sum(data)/len(data)/1e3], ind),
                    color = 'k', ls = '--', label = 'specified jitter')

            ax.set_title('Visual comparison of the measured two-channel jitters to specifications')
            ax.legend(loc = 1)
            plt.show()
            print('\n')

        print('Plotting a summary of the results')

        # Summary of the measured RMS jitters for a single channel in a table
        colLabels = ['single channel\nRMS jitter (ps)', 'within\nspecifications']
        fig, axes = plt.subplots(1, len(modes), figsize=(len(modes)*7, len(measured_channels[0])/1.6+0.7))
        axes = np.atleast_1d(axes)
        for i in np.arange(len(modes)):
            content = np.hstack((measured_jitters[i].reshape(-1, 1), within_specs[i].reshape(-1, 1)))
            colors = np.full_like(content, 'white')
            colors[within_specs[i] == True, 1] = 'C2'
            colors[within_specs[i] == False, 1] = 'C3'
            the_table = axes[i].table(cellText = content, colLabels = colLabels, rowLabels = measured_channels[i],
                                      loc = 'center', cellColours = colors, cellLoc = 'center')
            the_table.scale(0.78, 2.2)
            axes[i].axis('tight')
            axes[i].axis('off')
            if edition == 'High-Res':
                axes[i].set_title(list(self.res_modes.keys())[i], size = 12, pad = 7)
            else:
                axes[i].set_title(model + ' ' + edition, size = 12, pad = 7)
        axes[0].text(-0.05, 0.5, 'channel combination', rotation = 90, transform = axes[0].transAxes, va = 'center', size = 12)
        if edition == 'High-Res':
            figManager = plt.get_current_fig_manager()
            figManager.window.showMaximized()
        plt.show()

        return measured_jitters, within_specs, measured_channels



######################   measurement methods   ################################


    ###subclassing the CountRate measurement class
    def get_countrate(self, startfor = int(1E12), channels = [1, 2, 3, 4]):

        # With the TimeTaggerNetwork object, we can set up a measurement as usual
        self.rate = TimeTagger.Countrate(self.client, channels)

        self.rate.startFor(startfor)
        self.rate.waitUntilFinished()
        self.countrates = self.rate.getData()

        print('Measured count rates of channel 1-4 in counts/s:')
        print(self.countrates)
        return self.countrates

    def get_count(self, channels = [1, 2], binwidth = 1000, n = 1000):

        # With the TimeTaggerNetwork object, we can set up a measurement as usual
        self.count = TimeTagger.Counter(self.client, channels, binwidth, n)

        data = self.count.getData()
        print(np.shape(data))


        print('Measured count of channel 1-4 in counts:')
        print(data)
        plt.plot(data)
        plt.show()

        return data



    ###subclassing the timetagstream class which is under measurement classes
    ###buffer size is memory buffer allocated that is read and destroyed with each
    ###getData() call.
    ###the rate of getData() call must be quick relative to buffer size to avoid
    ###data overflow

    ###startfor is in picoseconds
    def streamdata(self, startfor = int(5E11), channels = [1, 2, 3, 4], buffer_size = 1000000, update_rate = -1.):

        format_string = '{:>8} | {:>17} | {:>7} | {:>14} | {:>13}'
        print(format_string.format('TAG #', 'EVENT TYPE', 'CHANNEL', 'TIMESTAMP (ps)', 'MISSED EVENTS'))
        print('---------+-------------------+---------+----------------+--------------')
        event_name = ['0 (TimeTag)', '1 (Error)', '2 (OverflowBegin)', '3 (OverflowEnd)', '4 (MissedEvents)']

        self.stream = TimeTagger.TimeTagStream(tagger = self.client,
                                          n_max_events = buffer_size,
                                          channels = channels)

        self.stream.startFor(startfor)
        event_counter = 0
        chunk_counter = 1

        while self.stream.isRunning():
            # getData() does not return timestamps, but an instance of TimeTagStreamBuffer
            # that contains more information than just the timestamp
            data = self.stream.getData()
            # print(data.getTimestamps())
            # print(len(data.getTimestamps()))
            # time.sleep(0.1)
            print(data.getEventTypes())
            if data.size:
                # With the following methods, we can retrieve a numpy array for the particular information:
                channel = data.getChannels()            # The channel numbers
                timestamps = data.getTimestamps()       # The timestamps in ps
                overflow_types = data.getEventTypes()   # TimeTag = 0, Error = 1, OverflowBegin = 2, OverflowEnd = 3, MissedEvents = 4
                missed_events = data.getMissedEvents()  # The numbers of missed events in case of overflow

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

        return format_string


if __name__ == '__main__':
    with TagClient() as tcee:
        print("\n\n################## With TagClient as tcee ###################\n\n")
        import code; code.interact(local=locals())
