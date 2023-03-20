#!/usr/bin/env python3
#%%


'''
This script provides the object class of a very specific setup:
a tektronix afg3000 waveform generator inputting a Thorlabs TEC 200C

In the specific context of modulating the FSR of a fibre cavity
to achieve a tunable bandpass filter 

Essentially we use the tektronix visa to talk to AFG but all functions
are in context of controlling TEC (ie DC voltage input to TEC)

Modify/use with other equipment AT YOUR OWN RISK

'''
import sys
import pyvisa

sys.path.append('../QMLab/experimental_control/')
import tektronix_func_gen as tfg

_VISA_ADDRESS = "USB0::0x0699::0x0353::1731975::INSTR"
_AFG3000_TCP_ADDRESS = 'TCPIP0::192.168.0.14::inst0::INSTR'

### please make channels an ordered list of integers....
class HotFibre():
    def __init__(self, sensor = 'TH20K', channels = [1]):

        self.afg = tfg.FuncGen(_AFG3000_TCP_ADDRESS)
        self.sensor = sensor
        self.volts_lower_bound = -5
        self.volts_upper_bound = 10
        self.channels = channels
        self.voltage_increment = 0.001
        self.kohm_increment = 0.002

        initial_voltage = 0.03

        self.voltages = [0. for i in range(len(channels))]
        self.resistances = [0. for i in range(len(channels))]

        if self.sensor ==  'TH20K':
            self.volt_kohm_coeff = 2 ###kilo ohms per volt!!!
            print('TH20K sensor selected! Coefficient is 2kohm/volt')

        elif self.sensor == 'TH200K':
            self.volt_kohm_coeff = 20
            print('TH200K sensor selected! Coefficient is 20kohm/volt')

        elif self.sensor == 'AD590/592' or self.sensor == 'LM135/335':
            self.volts_lower_bound = -2.25
            self.volts_upper_bound = 7.25
            self.celsius_volt_coeff = 20 ### celsius per volt
            print('AD590/592 or LM135/335 sensor selected! Coeffecient is 20c/volt')


        for channel in channels:
            self.set_afg_reset(channel)
            self.set_voltage_upper(channel, self.volts_upper_bound)
            self.set_voltage_lower(channel, self.volts_lower_bound)
            self.set_shape_dc(channel)
            self.set_voltage_dc(channel, initial_voltage)
            print('Channel {} set to {}V'.format(channel, initial_voltage))



    def set_resistance_kohms(self, channel, kohms):
        '''
        Set the temp parameter in kohms
        '''

        volts = kohms/self.volt_kohm_coeff
        bytes = self.set_voltage_dc(channel = channel, volts = volts) 
        self.resistances[channel-1] = kohms
        self.voltages[channel-1] = volts

        return bytes

    def set_voltage_dc(self, channel, volts):
        '''
        Set the temp parameter in volts
        '''
        bytes = self.afg.write('SOURce{}:VOLTage:LEVel:IMMediate:OFFSet {}V'.format(channel, volts))
        self.voltages[channel-1] = volts
        return bytes

    def set_afg_reset(self, channel):

        bytes1 = self.afg.write('SOURce{}:VOLTage:LEVel:IMMediate:OFFSet 0V'.format(channel))
        bytes2 = self.afg.write('*CLS')
        return [bytes1, bytes2]

    def set_voltage_upper(self, channel, volts):
        '''
        Set the upper limit of the voltage
        '''
        bytes = self.afg.write('SOURce{}:VOLTage:LIMit:HIGH {}V'.format(channel, volts))
        return bytes

    def set_voltage_lower(self, channel, volts):
        bytes = self.afg.write('SOURce{}:VOLTage:LIMit:LOW {}V'.format(channel, volts))
        return bytes

    def switchon_ch(self, channel):
        bytes = self.afg.write('OUTPut{}:STATe ON'.format(channel))
        return bytes 
    
    def switchoff_ch(self, channel):
        bytes = self.afg.write('OUTPut{}:STATe OFF'.format(channel))
        return bytes
    
    def set_shape_dc(self, channel):
        bytes = self.afg.write('SOURce{}:FUNCtion:SHAPe DC'.format(channel))
        return bytes

    def step_kohm(self, channel, step = 1):
        '''
        increments is an integer
        '''
        kohms = self.resistances[channel-1] - step*self.kohm_increment
        bytes = self.set_resistance_kohms(channel = channel, kohms = kohms)
        self.resistances[channel-1] = kohms
        return bytes
    
    def step_volts(self, channel, step = 1):
        '''
        increments is an integer
        '''
        volts = self.voltages[channel-1] + step*self.voltage_increment
        bytes = self.set_voltage_dc(channel = channel, volts = volts)
        self.voltages[channel-1] = volts
        print('Current voltage is', volts)
        return bytes

    def get_voltages(self):
        print('Current voltages are: {}V'.format(self.voltages))
        return self.voltages

    def get_resistances(self):
        print('Current resistances are: {}KOhms'.format(self.resistances))
        return self.resistances
    

    def __enter__(self):
        
        return self


    def __exit__(self, exception_type, exception_value, exception_traceback):
        pass


####################################################################################

if __name__ == '__main__':
    with HotFibre() as hf:
        print("\n\n################## With HotFibre as hf ###################\n\n")
        import code; code.interact(local=locals())
