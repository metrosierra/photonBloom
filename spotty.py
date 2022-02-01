#!/usr/bin/env python3

### interfacing script for timetagger object that is implicit with
### the photonSpot cryogenic nanowire photondetector
### the detector basically only sends analog signals directly to
### the timetagger timer box, so we can treat the timer and detector as one
### even though we are only using timetagger provided methods

### core assumption is that the detector gain is set: this should be done
### manually as a fail-safe against detector saturation which is CATASTROPHIC

### we want to include a devmode as well, so we create a wrapper class for the
### timetagger object that includes both timetagger and devmode methods



###%%%%%%%%%%%%%%%%%%%%%%
import TimeTagger as tt

import numpy as np



class taggerSpot():


    def __init__(self):

        print('\nTimetagger object initialising...assuming it is reading from PhotonSpot nanowire single-photon detector...will prompt about detector gain settings in a bit\n')

        print('\nAttempting handshake with timetagger control box\n')
        try:
            self.tagger = tt.createTimeTagger()
            print('Handshake shaken! Run printfunctions to get a list of ')


        except:
            print('Handshake failed! Defaulting to DevMode...testing functionalities limited.')


    def __enter__(self):
        return self

    def __exit__(self, exception_type, exception_value, exception_traceback):
        tt.freeTimeTagger(self.tagger)

print('hi')
###%%%%%%%%%%%%%%%%%%%%%%
