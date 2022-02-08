#!/usr/bin/env python3


###%%%%%%%%%%%%%%%%%%%%%%
import TimeTagger as tt
import numpy as np


from client_object import TagClient



class spotty():


    def __init__(self):

        print('Initiliasing prototype usage file powered by TagClient methods')


        self.spot0 = TagClient()

        input('please type in start_test into the server terminal session and press ENTER')
        print('attempting to stream data')
        self.spot0.streamdata(startfor = int(5E11), channels = [1, 2, 3, 4], n_max_events = 1000000):


    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass


###%%%%%%%%%%%%%%%%%%%%%%
