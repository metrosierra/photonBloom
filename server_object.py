#!/usr/bin/env python3

import sys
import TimeTagger
import socket

if sys.version_info < (3, 0):  # Python 2.7
    input = raw_input



class TagServer():


    def __init__(self):
        print('\nInitialising Server Object for timetagger (assuming the physical tagger is paired with this PC)\n')
        self.tagger = TimeTagger.createTimeTagger()

        print('\nInitialising test signals on all 4 channels...standby\n')
        # Connect to the Time Tagger and activate the internal test signal on four channels
        # self.set_testsignal()
        self.get_ip()


        # Start server with full control of the hardware by the connected clients.
        self.start_server()
        print('Time Tagger server started successfully on the default port 41101.\n')
        print('This PC is ready to be interfaced with via server as if it is a physical time tagger!!!!!!')

    def start_server(self):
        self.tagger.startServer(TimeTagger.AccessMode.Control)

    def get_ip(self):
        self.local_ip = socket.gethostbyname(socket.gethostname())
        print("Local IP address:\n{}\n".format(self.local_ip))

    def set_testsignal(self, channels = [1, 2, 3, 4]):
        self.tagger.setTestSignal(channels, True)

    def get_methods_naive(self):
        self.method_list = [attribute for attribute in dir(self) if callable(getattr(self, attribute)) and attribute.startswith('__') is False]

        return self.method_list



    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        tagger.stopServer()
        TimeTagger.freeTimeTagger(self.tagger)
        print('Ciao!')


if __name__ == '__main__':
    with TagServer() as tes:
        print("with TagServer as tes")
        # Keep the server alive until you press enter.
        input('Press ENTER to stop the server...')
