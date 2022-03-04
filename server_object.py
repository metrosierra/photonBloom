#!/usr/bin/env python3

import TimeTagger
import socket

from subroutines.mathematics import starsss, percentsss

class TagServer():


    def __init__(self, testsignal = False):


        self.welcome_message()
        self.istest = testsignal
        print('\nInitialising Server Object for timetagger (assuming the physical tagger is paired with this PC)\n')
        self.tagger = TimeTagger.createTimeTagger()

        print('\nInitialising test signals on all 4 channels...standby\n')
        # Connect to the Time Tagger and activate the internal test signal on four channels
        # self.set_testsignal()
        self.get_ip()

        if self.istest:
            self.set_testsignal(channels = [1, 2, 3 ,4])

        # Start server with full control of the hardware by the connected clients.
        self.start_server()
        print('Time Tagger server started successfully on the default port 41101.\n')
        print('This PC is ready to be interfaced with via server as if it is a physical time tagger!!!!!!')

    @percentsss 
    @starsss
    def welcome_message(self):
        print('Hello tout le monde!')
        

    def start_server(self):
        self.tagger.startServer(TimeTagger.AccessMode.Control)
        return self

    def get_ip(self):
        self.local_ip = socket.gethostbyname(socket.gethostname())
        print("Local IP address:\n{}\n".format(self.local_ip))

    def set_testsignal(self, channels = [1, 2, 3, 4]):

        if not self.istest:
            self.tagger.setTestSignal(channels, True)
            print('Test mode set! Activating test signals on channels {}'.format(channels))
            self.istest = True

        else:
            self.tagger.setTestSignal(channels, False)
            print('Test mode unset! Deactivating test signals on channels {}'.format(channels))
            self.istest = False

        return self


    def get_methods_naive(self):
        self.method_list = [attribute for attribute in dir(self) if callable(getattr(self, attribute)) and attribute.startswith('__') is False]
        return self.method_list


    def __enter__(self):
        return self

    @percentsss
    @starsss
    def __exit__(self, exception_type, exception_value, exception_traceback):
        self.tagger.stopServer()
        TimeTagger.freeTimeTagger(self.tagger)
        print('Ciao!')


if __name__ == '__main__':
    with TagServer() as tes:
        print("with TagServer as tes")

        alive = True
        while alive:
            # Keep the server alive until you press enter.
            keyinput = input('\n ########################## Press ENTER to stop the server and release timetagger... ##########################\n')
            if keyinput == 'set_test' and tes.istest == False:
                tes.set_testsignal(channels = [1, 2, 3 ,4])

            elif keyinput == 'stop_test' and tes.istest == True:
                tes.set_testsignal(channels = [1, 2, 3 ,4])

            elif keyinput == 'quit' or keyinput == '':
                alive = False
