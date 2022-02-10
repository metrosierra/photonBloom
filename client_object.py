#!/usr/bin/env python3

import sys
import TimeTagger
import socket

import numpy as np
import matplotlib.pyplot as plt
import time


class TagClient():


    def __init__(self, ip_address = '192.168.0.2'):

        self.target_ip = ip_address
        print("Search for Time Taggers on the network...")
        # Use the scanTimeTaggerServers() function to search for Time Tagger servers in the local network
        self.serverlist = TimeTagger.scanTimeTaggerServers()

        print("{} servers found.".format(len(self.serverlist)))
        print(self.serverlist)

        self.server_handshake()
        print('Server handshake successful!!!!!!!!!!!')
        print('\nTimetagger object initialising...assuming it is reading from PhotonSpot nanowire single-photon detector...will prompt about detector gain settings in a bit\n')
        self.get_methods_naive()
        print('Here are the available class methods to be used in interactive mode')

    def server_handshake(self):
        try:
            self.target_server_info = TimeTagger.getTimeTaggerServerInfo('{}'.format(self.target_ip))
            print('Information about Time Tagger server on {}:'.format(self.target_ip))
            print(self.target_server_info)

        except RuntimeError:
            raise Exception('No Time Tagger server available on {} and the default port 41101.'.format(self.target_ip))

        self.client = TimeTagger.createTimeTaggerNetwork(self.target_ip)
        print('Connecting to the server on localhost.')

    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        TimeTagger.freeTimeTagger(self.client)
        print('Ciao!')

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
