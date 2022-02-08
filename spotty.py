#!/usr/bin/env python3


###%%%%%%%%%%%%%%%%%%%%%%
import TimeTagger as tt
import numpy as np

import matplotlib.pyplot as plt

from client_object import TagClient



class spotty():


    def __init__(self):

        print('Initiliasing prototype usage file powered by TagClient methods')


        self.spot0 = TagClient('192.168.0.2')

        input('please type in start_test into the server terminal session and press ENTER')
        print('attempting to stream data')
        data = self.spot0.get_count()
        plt.plot(data)
        plt.show()
        #spot0.histogram(aowdaowdaowdbaowdbaa)




    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass

hi = spotty()
###%%%%%%%%%%%%%%%%%%%%%%
